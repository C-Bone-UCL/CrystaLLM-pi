"""API endpoint tests: training."""

import os
import json

class TrainingEndpointTests:
    """Test training endpoints."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_train_single_gpu(self):
        """Test train with single GPU config."""
        # create minimal config
        config = {
            "model_name": "test-model",
            "huggingface_dataset": "c-bone/SLME_1K_dtest"
        }
        config_path = os.path.join(self.temp_dir, "test_train.jsonc")
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        response = self.client.post("/train", json={
            "config_file": config_path,
            "multi_gpu": False
        })
        assert response.status_code == 200
        data = response.json()
        assert "python" in data["command"]
        assert "_train" in data["command"]
        assert "torchrun" not in data["command"]
        
    def test_train_multi_gpu(self):
        """Test train with multi-GPU config."""
        config_path = os.path.join(self.temp_dir, "test_train_multi.jsonc")
        with open(config_path, 'w') as f:
            json.dump({"model_name": "test"}, f)
            
        response = self.client.post("/train", json={
            "config_file": config_path,
            "multi_gpu": True,
            "nproc_per_node": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert "torchrun" in data["command"]
        assert "--nproc_per_node=2" in data["command"]
