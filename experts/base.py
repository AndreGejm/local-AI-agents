from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..llm_client import run_ollama_api

class BaseExpert(ABC):
    """Base class for all local specialist experts."""
    
    def __init__(self, name: str, phase_name: str):
        self.name = name
        self.phase_name = phase_name

    @abstractmethod
    async def execute(self, task: str, context: str, timeout_sec: int) -> Dict[str, Any]:
        """Execute the expert's task."""
        pass

    def _validate_output(self, output: str, required_markers: List[str]) -> bool:
        """Check if output contains all required markers."""
        output_lower = output.lower()
        for marker in required_markers:
            if marker.lower() not in output_lower:
                return False
        return True
