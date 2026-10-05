from pydantic import BaseModel, Field, field_validator

# Class coding used by every saved file, the calibration and the classifier output:
# 0 Left Hand, 1 Rest, 2 Right Hand. Defined here once; do not re-declare it elsewhere.
CLASS_ORDER = ("Left Hand", "Rest", "Right Hand")
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_ORDER)}


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
    model_path: str = "models/eegnet_finetuned_mimed.pth"
    step_samples: int = 64      
    n_outputs: int = 3   
    confidence_threshold: float = 0.5
    classes: list[str] = list(CLASS_ORDER)
    colors: dict[str, str] = {
        "Left Hand":  "#4169E1",
        "Rest":       "#2E8B57",
        "Right Hand": "#DC143C"
    }

class RecordingConfig(BaseModel):
    trial_seconds: float = 4.0
    prepare_seconds: float = 1.0
    rest_seconds: float = 2.0
    break_seconds: int = 30
    n_blocks: int = 6
    trials_per_block: int = 10
    labels: list[str] = list(CLASS_ORDER)
    colors: dict[str, str] = {
        "Left Hand":  "#4169E1",
        "Right Hand": "#DC143C",
        "Rest":       "#2E8B57",
    }
    save_path: str = "recordings"
    simulation_mode: bool = True

    @field_validator("labels")
    @classmethod
    def _labels_follow_class_order(cls, v):
        if tuple(v) != CLASS_ORDER:
            raise ValueError(f"labels must be {list(CLASS_ORDER)} (saved label codes depend on it), got {v}")
        return v

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