import numpy as np
from typing import Tuple


class Aggregator:
    def __init__(self, kind: str, crt_thresh: float = 0.0):
        """
        Aggregator object constructor.

        Returns
        -------
        output : Aggregator object

        Parameters
        ----------
        kind : string
            Aggregation function.
        crt_thresh : float, optional
            Use only data with cloud / certainty mask > crt_thresh / 100 %.
        """
        self.vi_a = None
        self.crt_a = None
        if kind not in ("avg", "min", "max", "sum"):
            raise Exception("Unknown aggregation type {}".format(kind))
        self.kind = kind
        self.crt_thresh = crt_thresh
        self.cnt = 0

    def push(self, vi: np.ndarray, crt: np.ndarray):
        """
        Push an image for aggregation.

        Returns
        -------
        output : None

        Parameters
        ----------
        vi : np.ndarray
            A single channel with values [0..1] float32.
        crt : np.ndarray
            Cloud / certainty mask.
        """
        if vi is None:
            print("Got None VI for aggregation")
            return

        if crt is None:
            print("Got None CRT for aggregation")
            return

        crt_mask = crt > 0
        if not self.cnt:
            self.vi_a = np.zeros_like(vi, dtype=vi.dtype)
            if self.kind in ["avg", "sum"]:
                self.vi_a_sqr = np.zeros_like(vi, dtype=vi.dtype)
            if self.kind == "min":
                np.add(self.vi_a, 1.0, out=self.vi_a)
            self.crt_a = np.zeros_like(vi, dtype=vi.dtype)

        if self.kind in ["avg", "sum"]:
            crt_coverage = crt_mask.mean()
            if crt_coverage > self.crt_thresh:
                self.vi_a += vi * crt_mask
                self.vi_a_sqr += (vi ** 2) * crt_mask
                self.crt_a += crt_mask
                self.cnt += 1
        else:
            self.vi_a[crt_mask] = getattr(self, "_" + self.kind)(
                self.vi_a, vi, crt_mask
            )
            self.cnt += 1

    def _min(
        self, acc: np.ndarray, a: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        return np.minimum(acc[mask], a[mask])

    def _max(
        self, acc: np.ndarray, a: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        return np.maximum(acc[mask], a[mask])

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregation results getter.

        Returns
        -------
        output : tuple
            Tuple of np.ndarrays: 3 for avg func, 2 for others.

        Notes
        -----
        example:
        >> a = Aggregator('avg', 0.5)
        >> for ndvi, cloud_mask in zip(ndvi_imgs, cloud_masks):
        ...  a.push(ndvi, cloud_mask)
        >> ndvi_avg, ndvi_std, clouds_avg = a.get()
        >> b = Aggregator('max')
        >> for msavi2, cloud_mask in zip(msavi2_imgs, cloud_masks):
        ...  b.push(msavi2, cloud_mask)
        >> msavi2_max = a.get()[0]
        """
        if self.vi_a is None or self.crt_a is None:
            raise Exception("No data to aggregate")
        if self.kind == "avg":
            avg_vi = self.vi_a / self.crt_a
            avg_vi_sqr = self.vi_a_sqr / self.crt_a
            avg_crt = self.crt_a / self.cnt
            avg_vi[~np.isfinite(avg_vi)] = 0
            avg_vi_sqr[~np.isfinite(avg_vi_sqr)] = 0
            avg_crt[~np.isfinite(avg_crt)] = 0
            return (avg_vi, np.sqrt(avg_vi_sqr - avg_vi ** 2), avg_crt)
        else:
            return (self.vi_a, self.crt_a)
