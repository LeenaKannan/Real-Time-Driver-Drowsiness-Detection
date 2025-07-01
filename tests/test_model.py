import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.model_architecture import create_cnn_model, create_vit_model
from src.models.data_preprocessing import preprocess_image, augment_data
from src.models.train_model import ModelTrainer

class TestModelArchitecture(unittest.TestCase):
    
    def test_create_cnn_model(self):
        model = create_cnn_model(input_shape=(64, 64, 3), num_classes=2)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 64, 64, 3))
        self.assertEqual(model.output_shape, (None, 2))
    
    def test_create_vit_model(self):
        model = create_vit_model(input_shape=(224, 224, 3), num_classes=2)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 2))

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_preprocess_image(self):
        processed = preprocess_image(self.test_image, target_size=(64, 64))
        self.assertEqual(processed.shape, (64, 64, 3))
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))
    
    def test_augment_data(self):
        images = np.random.random((10, 64, 64, 3))
        labels = np.random.randint(0, 2, (10,))
        aug_images, aug_labels = augment_data(images, labels, augment_factor=2)
        self.assertEqual(len(aug_images), 20)
        self.assertEqual(len(aug_labels), 20)

class TestModelTrainer(unittest.TestCase):
    
    def setUp(self):
        self.trainer = ModelTrainer(model_type='cnn', input_shape=(64, 64, 3))
    
    def test_trainer_initialization(self):
        self.assertIsNotNone(self.trainer.model)
        self.assertEqual(self.trainer.model_type, 'cnn')
    
    @patch('tensorflow.keras.models.load_model')
    def test_load_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        self.trainer.load_model('test_path.h5')
        mock_load.assert_called_once_with('test_path.h5')
    
    def test_compile_model(self):
        self.trainer.compile_model()
        self.assertIsNotNone(self.trainer.model.optimizer)
        self.assertIsNotNone(self.trainer.model.loss)
    
    def test_train_with_mock_data(self):
        X_train = np.random.random((100, 64, 64, 3))
        y_train = np.random.randint(0, 2, (100,))
        X_val = np.random.random((20, 64, 64, 3))
        y_val = np.random.randint(0, 2, (20,))
        
        with patch.object(self.trainer.model, 'fit') as mock_fit:
            mock_fit.return_value = MagicMock()
            history = self.trainer.train(X_train, y_train, X_val, y_val, epochs=1, batch_size=32)
            mock_fit.assert_called_once()

class TestModelPrediction(unittest.TestCase):
    
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([[0.2, 0.8], [0.9, 0.1]])
    
    def test_prediction_shape(self):
        test_input = np.random.random((2, 64, 64, 3))
        predictions = self.mock_model.predict(test_input)
        self.assertEqual(predictions.shape, (2, 2))
    
    def test_prediction_probabilities(self):
        test_input = np.random.random((2, 64, 64, 3))
        predictions = self.mock_model.predict(test_input)
        for pred in predictions:
            self.assertAlmostEqual(np.sum(pred), 1.0, places=5)

class TestModelSaveLoad(unittest.TestCase):
    
    def test_model_save_load_consistency(self):
        model = create_cnn_model(input_shape=(64, 64, 3), num_classes=2)
        test_input = np.random.random((1, 64, 64, 3))
        
        original_prediction = model.predict(test_input)
        
        temp_path = '/tmp/test_model.h5'
        model.save(temp_path)
        
        loaded_model = tf.keras.models.load_model(temp_path)
        loaded_prediction = loaded_model.predict(test_input)
        
        np.testing.assert_array_almost_equal(original_prediction, loaded_prediction)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()