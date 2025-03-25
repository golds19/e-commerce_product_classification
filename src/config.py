from pathlib import Path

class Config:
    # Project structure
    ROOT_DIR = Path(__file__).parent.parent  # Gets the project root directory
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "models"
    LOG_DIR = ROOT_DIR / "logs"

    # Dataset parameters
    DATASET_PATH = DATA_DIR  # Update this with your actual path
    IMAGE_SIZE = (224, 224)
    VALIDATION_SPLIT = 0.2
    CHOSEN_SUBCATEGORIES = [
        "Topwear",
        "Shoes",
        "Bottomwear",
        "Headwear",
        "Sports Accessories"
    ]

    # DataLoader parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Model parameters
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = len(CHOSEN_SUBCATEGORIES)
    PRETRAINED = True
    FREEZE_BACKBONE = True
    DROPOUT_RATE = 0.2

    # Training parameters
    DEVICE = "cuda"  # or "cpu"
    EPOCHS = 5
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Optimizer parameters
    OPTIMIZER = {
        "name": "Adam",
        "params": {
            "lr": LEARNING_RATE,
        }
    }

    # Data augmentation parameters
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet means
    NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet stds
    
    # Training augmentations
    # TRAIN_AUGMENTATION = {
    #     "random_horizontal_flip": 0.5,  # probability
    #     "random_rotation": 15,         # degrees
    #     "random_brightness": 0.2,      # brightness factor
    #     "random_contrast": 0.2,        # contrast factor
    # }

    # Logging parameters
    SAVE_MODEL_FREQUENCY = 1  # Save model every N epochs
    LOG_FREQUENCY = 100       # Log metrics every N batches
    
    # Early stopping parameters
    EARLY_STOPPING = {
        "patience": 5,
        "min_delta": 0.001
    }

    # Checkpoint parameters
    CHECKPOINT = {
        "save_best_only": True,
        "save_frequency": 1
    }

    # Random seed for reproducibility
    RANDOM_SEED = 42


class DevelopmentConfig(Config):
    """Configuration for development environment"""
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    EPOCHS = 2


class ProductionConfig(Config):
    """Configuration for production environment"""
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    PIN_MEMORY = True


# Select the configuration to use
config = Config  # or DevelopmentConfig or ProductionConfig