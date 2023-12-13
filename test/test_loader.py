import pytest
import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(os.path.join("/home/lorenzo-sibi/Scrivania/research_project"))
from src import loader, utils
from src.utils import TensorContainer, TensorType

TEST_REAL_IMAGES_FOLDER = Path(Path(__file__).parent, "test_images", "real")
TEST_FAKE_IMAGES_FOLDER = Path(Path(__file__).parent, "test_images", "fake")

TEST_REAL_TENSORS_FOLDER = Path(Path(__file__).parent, "test_latent_spaces")


class TestLoaderClass():
    
    def test_load_images_as_list(self):
        
        filenames = [filename for filename in TEST_REAL_IMAGES_FOLDER.iterdir()]
        
        try:
            test_real_images = [Image.open(filename) for filename in filenames]
            test_fake_images = [Image.open(filename) for filename in filenames]

            real_images = loader.load_images_as_list(TEST_REAL_IMAGES_FOLDER)
            fake_images = loader.load_images_as_list(TEST_REAL_IMAGES_FOLDER)
            
            for (t_real, real, t_fake, fake) in zip(test_real_images, real_images, test_fake_images, fake_images):
                assert t_real == real
                assert t_fake == fake
            
        except Exception as e:
            print(e)
            assert False

    def test_load_tensors_from_directory(self):
        
        test_real_tensors = []
        
        real_tensors = loader.load_tensors_from_directory(TEST_REAL_TENSORS_FOLDER)
        
        for t_r_filename in TEST_REAL_TENSORS_FOLDER.iterdir():
            if t_r_filename.is_dir():
                continue
            
            name = t_r_filename.stem
            
            with np.load(t_r_filename) as data:
                for _, item in data.items():
                    test_real_tensors.append(TensorContainer(item, name, TensorType.NP_TENSOR))
        
        for (t_real, real) in zip(test_real_tensors, real_tensors):
            assert np.array_equal(t_real.tensor, real.tensor)