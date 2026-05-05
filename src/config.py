from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

env_file_path = Path(__file__).parent.parent / '.env'

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(env_file_path), case_sensitive=True, extra="ignore")

    HF_TOKEN: str = ''

settings = Settings()