import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmengine.registry import TRANSFORMS
from typing import Optional


@TRANSFORMS.register_module()
class LoadCDImagesFromFile(LoadImageFromFile):
    """Load both images from file for change detection datasets.
    
    This transform loads both img1 and img2 from img1_path and img2_path
    used in change detection datasets.
    
    Note: We inherit from mmcv's LoadImageFromFile to reuse its initialization
    parameters (color_type, channel_order, etc.), but we override transform()
    to load both images simultaneously.
    """
    
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load both images.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """
        
        def _load_image(filename: str) -> np.ndarray:
            """Helper function to load a single image."""
            try:
                if self.file_client_args is not None:
                    file_client = fileio.FileClient.infer_client(
                        self.file_client_args, filename)
                    img_bytes = file_client.get(filename)
                else:
                    img_bytes = fileio.get(
                        filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e
            # in some cases, images are not read successfully, the img would be
            # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
            assert img is not None, f'failed to load image: {filename}'
            if self.to_float32:
                img = img.astype(np.float32)
            return img

        # Load img1
        img1 = _load_image(results['img1_path'])
        if img1 is None:
            return None
        
        results['img'] = img1
        results['img_shape'] = img1.shape[:2]
        results['ori_shape'] = img1.shape[:2]
        
        # Load img2
        img2 = _load_image(results['img2_path'])
        if img2 is None:
            return None
        
        results['img2'] = img2
        results['img2_shape'] = img2.shape[:2]
        results['img2_ori_shape'] = img2.shape[:2]
        
        return results
