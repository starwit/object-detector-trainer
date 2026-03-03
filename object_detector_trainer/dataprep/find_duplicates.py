from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import Dict, List, Tuple, Set, NamedTuple

import cv2
import imagehash
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class Match(NamedTuple):
    """Similarity match for one image against a target image."""
    target: Path
    ssim_score: float
    hamming: int


class DisjointSet:
    """
    DisjointSet (Union-Find) data structure for efficiently grouping elements into disjoint sets.
    Used in the image duplicate detection to cluster similar images together.

    Key Operations:
    - find(): Determines which set an element belongs to
    - union(): Merges two sets together

      # Example: We have images: A, B, C
      # And we found these are similar:
      # A similar to B
      # B similar to C

      # Initially:
      # A → A
      # B → B
      # C → C

      ds.union('A', 'B')  # Now B points to A
      # A → A
      # B → A
      # C → C

      ds.union('B', 'C')  # Now C points to A (through B)
      # A → A
      # B → A
      # C → A

      # Now if we call find() on any image, it returns the root:
      ds.find('C')  # Returns 'A'

      # This creates cluster:
      # Cluster 1: A, B, C (with A as root)
    """

    def __init__(self):
        self.parent: Dict[Path, Path] = {}

    # -- core operations -----------------------------------------------------

    def find(self, item):
        """
        Find the root/representative element of the set containing 'item'.
        """
        # If item isn't in any set yet, create a new set with item as its own parent
        if item not in self.parent:
            self.parent[item] = item
        # Path compression
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        """
        Merge the sets containing item1 and item2.
        After union, both items will be in the same set.
        """
        # Find the roots of both items
        root1 = self.find(item1)
        root2 = self.find(item2)

        # If items are in different sets (have different roots) merge them
        # by making root2 point to root1
        if root1 != root2:
            self.parent[root2] = root1


class DuplicateDetector:
    """Detect duplicate / near-duplicate images using a two-stage pipeline.

    1. **Perceptual hash (pHash)** – cheap fingerprint; small Hamming distance
       marks candidates.
    2. **SSIM** – more expensive pixel-domain metric; confirms similarity.
    """

    def __init__(self, phash_threshold: int = 2, ssim_threshold: float = 0.95):
        self.phash_threshold = phash_threshold  # max acceptable Hamming dist
        self.ssim_threshold = ssim_threshold    # min acceptable SSIM score

    # --------------- Hash & similarity helpers ----------------

    @staticmethod
    def compute_perceptual_hash(file_path: Path) -> Tuple[Path, str | None]:
        """Compute perceptual hash; return `(path, hash_hex_or_None)`."""
        try:
            with Image.open(file_path) as img:
                return file_path, str(imagehash.phash(img))
        except (OSError, UnidentifiedImageError, ValueError):
            return file_path, None  # unreadable / unsupported image

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Cheap Hamming distance via XOR."""
        return (hash1 ^ hash2).bit_count()

    @staticmethod
    def compute_ssim(image1_path: Path, image2_path: Path) -> float:
        """Compute SSIM on 256×256 grayscale thumbnails."""
        try:
            img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                return 0.0
            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))
            score, _ = ssim(img1, img2, full=True)
            return score
        except (cv2.error, OSError, ValueError):
            return 0.0


    def find_duplicates(self, image_paths: List[Path]) -> Dict[Path, List[Path]]:
        """Return `{cluster_root: [img1, img2, …]}` for all duplicate clusters."""
        # 1. Compute pHashes in parallel (releases GIL)
        try:
            with ProcessPoolExecutor() as pool:
                hash_results = list(pool.map(self.compute_perceptual_hash, image_paths))
        except (OSError, PermissionError) as exc:
            # Some restricted runtimes disallow process-based multiprocessing (semaphores).
            logger.warning("Falling back to in-process hashing due to multiprocessing error: %s", exc)
            hash_results = [self.compute_perceptual_hash(path) for path in image_paths]

        # Keep successful hashes and convert hex→int once
        int_hashes: List[Tuple[Path, int]] = [
            (path, int(h, 16)) for path, h in hash_results if h is not None
        ]

        ds = DisjointSet()

        # 2. Exhaustive pairwise comparison (upper-triangle)
        for idx, (path_i, hash_i) in enumerate(int_hashes):
            for path_j, hash_j in int_hashes[idx + 1 :]:
                # cheap candidate filter
                if self.hamming_distance(hash_i, hash_j) > self.phash_threshold:
                    continue
                # expensive confirmation
                if self.compute_ssim(path_i, path_j) >= self.ssim_threshold:
                    ds.union(path_i, path_j)

        # 3. Collect clusters
        clusters: Dict[Path, List[Path]] = defaultdict(list)
        for path, _ in int_hashes:
            clusters[ds.find(path)].append(path)

        # keep only duplicate groups
        return {root: imgs for root, imgs in clusters.items() if len(imgs) > 1}

    def compare_folders(
        self, folder1: Path, folder2: Path
    ) -> Dict[Path, List[Match]]:
        """Compare images between two folders and report similar pairs."""
        def collect_images(folder: Path) -> List[Path]:
            return (
                list(folder.rglob("*.jpg"))
                + list(folder.rglob("*.jpeg"))
                + list(folder.rglob("*.png"))
            )

        folder1_images = collect_images(folder1)
        folder2_images = collect_images(folder2)

        try:
            with ProcessPoolExecutor() as pool:
                folder1_hashes = list(pool.map(self.compute_perceptual_hash, folder1_images))
                folder2_hashes = list(pool.map(self.compute_perceptual_hash, folder2_images))
        except (OSError, PermissionError) as exc:
            logger.warning("Falling back to in-process hashing due to multiprocessing error: %s", exc)
            folder1_hashes = [self.compute_perceptual_hash(path) for path in folder1_images]
            folder2_hashes = [self.compute_perceptual_hash(path) for path in folder2_images]

        folder1_hashes = [(p, int(h, 16)) for p, h in folder1_hashes if h]
        folder2_hashes = [(p, int(h, 16)) for p, h in folder2_hashes if h]

        matches: Dict[Path, List[Match]] = defaultdict(list)
        for path1, hash1 in folder1_hashes:
            for path2, hash2 in folder2_hashes:
                hamming = self.hamming_distance(hash1, hash2)
                if hamming <= self.phash_threshold:
                    ssim_score = self.compute_ssim(path1, path2)
                    if ssim_score >= self.ssim_threshold:
                        matches[path1].append(
                            Match(path2, ssim_score, hamming)
                        )
        return matches

    def print_folder_comparison_results(
        self, matches: Dict[Path, List[Match]]
    ) -> None:
        if not matches:
            print("\nNo similar images found between the folders.")
            return
        print("\nFound similar images between folders:")
        for i, (src, sims) in enumerate(matches.items(), start=1):
            print(f"\nImage {i} from source folder:")
            print(f"  Source: {src}")
            print("  Similar images in target folder:")
            for m in sims:
                print(f"    - {m.target}")
                print(f"      SSIM: {m.ssim_score:.3f}, Hamming distance: {m.hamming}")

    def print_duplicate_clusters(self, clusters: Dict[Path, List[Path]]) -> None:
        print(f"\nFound {len(clusters)} duplicate clusters:")
        for i, (root, imgs) in enumerate(clusters.items(), start=1):
            print(f"\nCluster {i}:")
            for img in imgs:
                print(f"  • {img}")

    def get_unique_images(self, image_paths: List[Path]) -> Set[Path]:
        """Return the set of unique images (one representative per cluster)."""
        clusters = self.find_duplicates(image_paths)
        duplicates: Set[Path] = set()
        for imgs in clusters.values():
            keep = min(imgs)               # keep lexicographically 1st
            duplicates.update(set(imgs) - {keep})
        return set(image_paths) - duplicates
