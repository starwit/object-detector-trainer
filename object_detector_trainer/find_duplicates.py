from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import cv2
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim


class Match(NamedTuple):
    """Similarity match for one image against a target image."""
    target: Path
    ssim_score: float
    hamming: int


class DisjointSet:
    """Union-Find for grouping duplicate images into clusters."""

    def __init__(self):
        self.parent: dict[Path, Path] = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
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
    def compute_perceptual_hash(file_path: Path) -> tuple[Path, str]:
        """Compute perceptual hash for one image."""
        with Image.open(file_path) as img:
            return file_path, str(imagehash.phash(img))

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Cheap Hamming distance via XOR."""
        return (hash1 ^ hash2).bit_count()

    @staticmethod
    def compute_ssim(image1_path: Path, image2_path: Path) -> float:
        """Compute SSIM on 256×256 grayscale thumbnails."""
        img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise FileNotFoundError(
                f"Could not read images for SSIM: {image1_path}, {image2_path}"
            )
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        score, _ = ssim(img1, img2, full=True)
        return float(score)


    def find_duplicates(self, image_paths: list[Path]) -> dict[Path, list[Path]]:
        """Return `{cluster_root: [img1, img2, …]}` for all duplicate clusters."""
        with ProcessPoolExecutor() as pool:
            hash_results = list(pool.map(self.compute_perceptual_hash, image_paths))

        # Keep successful hashes and convert hex→int once
        int_hashes = [(path, int(h, 16)) for path, h in hash_results]

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
        clusters: dict[Path, list[Path]] = defaultdict(list)
        for path, _ in int_hashes:
            clusters[ds.find(path)].append(path)

        # keep only duplicate groups
        return {root: imgs for root, imgs in clusters.items() if len(imgs) > 1}

    def compare_folders(
        self, folder1: Path, folder2: Path
    ) -> dict[Path, list[Match]]:
        """Compare images between two folders and report similar pairs."""
        def collect_images(folder: Path) -> list[Path]:
            return (
                list(folder.rglob("*.jpg"))
                + list(folder.rglob("*.jpeg"))
                + list(folder.rglob("*.png"))
            )

        folder1_images = collect_images(folder1)
        folder2_images = collect_images(folder2)

        with ProcessPoolExecutor() as pool:
            folder1_hashes = list(pool.map(self.compute_perceptual_hash, folder1_images))
            folder2_hashes = list(pool.map(self.compute_perceptual_hash, folder2_images))

        folder1_hashes = [(p, int(h, 16)) for p, h in folder1_hashes]
        folder2_hashes = [(p, int(h, 16)) for p, h in folder2_hashes]

        matches: dict[Path, list[Match]] = defaultdict(list)
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
        self, matches: dict[Path, list[Match]]
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

    def print_duplicate_clusters(self, clusters: dict[Path, list[Path]]) -> None:
        print(f"\nFound {len(clusters)} duplicate clusters:")
        for i, (root, imgs) in enumerate(clusters.items(), start=1):
            print(f"\nCluster {i}:")
            for img in imgs:
                print(f"  • {img}")

    def get_unique_images(self, image_paths: list[Path]) -> set[Path]:
        """Return the set of unique images (one representative per cluster)."""
        clusters = self.find_duplicates(image_paths)
        duplicates: set[Path] = set()
        for imgs in clusters.values():
            keep = min(imgs)               # keep lexicographically 1st
            duplicates.update(set(imgs) - {keep})
        return set(image_paths) - duplicates
