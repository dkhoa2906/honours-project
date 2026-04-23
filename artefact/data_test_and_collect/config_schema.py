from pydantic import BaseModel, Field

class CortexAPIConfig(BaseModel):
    client_id: str
    client_secret: str

class PreprocessingConfig(BaseModel):
    sampling_rate: int = 128

class ModelConfig(BaseModel):
    n_channels: int = 14
    buffer_seconds: float = 4.0

class LiveConfig(BaseModel):
    simulation_mode: bool  = True
    model_path: str = "models/eegnet_finetuned.pth"
    step_samples: int = 64      
    n_outputs: int = 3   
    confidence_threshold: float = 0.5
    classes: list[str] = ["Left Hand", "Rest", "Right Hand"]
    colors: dict[str, str] = {
        "Left Hand":  "#4169E1",
        "Rest":       "#2E8B57",
        "Right Hand": "DC143C"
    }

class RecordingConfig(BaseModel):
    trial_seconds: float = 4.0
    prepare_seconds: float = 1.0
    rest_seconds: float = 2.0
    break_seconds: int = 30
    n_blocks: int = 6
    trials_per_block: int = 10
    labels: list[str] = ["Left Hand", "Rest", "Right Hand"]
    colors: dict[str, str] = {
        "Left Hand":  "#4169E1",
        "Right Hand": "#DC143C",
        "Rest":       "#2E8B57",
    }
    save_path: str = "recordings"
    simulation_mode: bool = True

class AppConfig(BaseModel):
    cortex_api: CortexAPIConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)

class DataCollectConfig(BaseModel):
    cortex_api: CortexAPIConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)