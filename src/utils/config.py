"""
Module for loading and validating iText2KG configuration from TOML file.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# TOML support for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "tomli library is required for Python < 3.11. Install it with: pip install tomli>=2.0.0"
        )


class ConfigValidationError(Exception):
    """Exception for configuration validation errors."""

    pass


def _provider_of(section: Dict[str, Any]) -> str:
    """Return normalized LLM provider name for a config section.

    Defaults to 'openai' when the field is missing for backward compatibility.
    """
    return str(section.get("provider", "openai")).lower().strip() or "openai"


def _placeholder_key(key: str) -> bool:
    """True if api_key is missing, whitespace-only, or an unfilled placeholder."""
    if not key or not key.strip():
        return True
    stripped = key.strip()
    return stripped.startswith("sk-...") or stripped.startswith("AIza...")


def _inject_env_api_keys(config: Dict[str, Any]) -> None:
    """
    Injects API keys from environment variables.

    For each LLM-using section ([itext2kg_concepts], [itext2kg_graph], [refiner])
    the env var picked depends on the section's `provider` field:
      provider = "openai" (default)  -> OPENAI_API_KEY
      provider = "gemini"            -> GEMINI_API_KEY

    Embedding sections ([dedup], [refiner].embedding_api_key) always use
    OPENAI_EMBEDDING_API_KEY (falling back to OPENAI_API_KEY) for now;
    Gemini embeddings will land in a follow-up PR.

    Priority:
    1. Environment variable (if set)
    2. Value from config.toml (if not placeholder)
    3. Validation error in _validate_*_section
    """
    env_openai = os.getenv("OPENAI_API_KEY")
    env_gemini = os.getenv("GEMINI_API_KEY")

    for section_name in ("itext2kg_concepts", "itext2kg_graph", "refiner"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        provider = _provider_of(section)
        env_key = env_gemini if provider == "gemini" else env_openai
        if env_key and _placeholder_key(section.get("api_key", "")):
            section["api_key"] = env_key

    # Embedding API keys (can use separate key).
    # Embeddings stay on OpenAI for now; Gemini embeddings land in a follow-up PR.
    env_embedding_key = os.getenv("OPENAI_EMBEDDING_API_KEY", env_openai)

    # dedup.embedding_api_key
    if env_embedding_key:
        if "dedup" in config:
            current_key = config["dedup"].get("embedding_api_key", "")
            if not current_key or current_key.startswith("sk-..."):
                config["dedup"]["embedding_api_key"] = env_embedding_key

    # refiner.embedding_api_key
    if env_embedding_key:
        if "refiner" in config:
            current_key = config["refiner"].get("embedding_api_key", "")
            if not current_key or current_key.startswith("sk-..."):
                config["refiner"]["embedding_api_key"] = env_embedding_key


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads and validates configuration from TOML file.

    Args:
        config_path: Path to configuration file.
                    If None, uses src/config.toml

    Returns:
        Dictionary with validated configuration

    Raises:
        ConfigValidationError: On validation errors
        FileNotFoundError: If configuration file not found
    """
    if config_path is None:
        # Determine path to config.toml relative to this file
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config.toml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load TOML file
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        raise ConfigValidationError(f"Failed to parse TOML file: {e}")

    # Check if this is a viz config (has viz-specific sections)
    is_viz_config = "graph2metrics" in config or "visualization" in config

    # Only inject API keys and validate main sections for non-viz configs
    if not is_viz_config:
        # Inject API keys from env variables
        _inject_env_api_keys(config)

        # Validate configuration
        try:
            _validate_config(config)
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")

    # Validate is_reasoning parameter is present (only for main pipeline)
    if "itext2kg_concepts" in config:
        if "is_reasoning" not in config["itext2kg_concepts"]:
            raise ConfigValidationError(
                "Parameter 'is_reasoning' is required in [itext2kg_concepts] section"
            )

    if "itext2kg_graph" in config:
        if "is_reasoning" not in config["itext2kg_graph"]:
            raise ConfigValidationError(
                "Parameter 'is_reasoning' is required in [itext2kg_graph] section"
            )

    if "refiner" in config:
        if "is_reasoning" not in config["refiner"]:
            raise ConfigValidationError("Parameter 'is_reasoning' is required in [refiner] section")

    # Optional consistency check (warning, not error)
    import logging

    logger = logging.getLogger(__name__)
    for section in ["itext2kg_concepts", "itext2kg_graph", "refiner"]:
        if section in config:
            is_reasoning = config[section].get("is_reasoning", False)
            has_temperature = config[section].get("temperature") is not None
            has_reasoning_effort = config[section].get("reasoning_effort") is not None

            if is_reasoning and has_temperature:
                logger.warning(
                    f"[{section}] Reasoning model with temperature parameter - "
                    f"might be ignored by API"
                )
            if not is_reasoning and has_reasoning_effort:
                logger.warning(
                    f"[{section}] Non-reasoning model with reasoning_effort - "
                    f"will be ignored by API"
                )

    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Валидирует полную структуру конфигурации."""
    required_sections = ["slicer", "itext2kg_concepts", "itext2kg_graph", "dedup", "refiner"]

    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: [{section}]")

    # Валидируем каждую секцию
    _validate_slicer_section(config["slicer"])
    _validate_itext2kg_concepts_section(config["itext2kg_concepts"])
    _validate_itext2kg_graph_section(config["itext2kg_graph"])
    _validate_dedup_section(config["dedup"])
    _validate_refiner_section(config["refiner"])


def _validate_slicer_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [slicer]."""
    required_fields = {
        "max_tokens": int,
        "soft_boundary": bool,
        "soft_boundary_max_shift": int,
        "tokenizer": str,
        "allowed_extensions": list,
    }

    _validate_required_fields(section, required_fields, "slicer")

    # Проверяем диапазоны значений
    if section["max_tokens"] <= 0:
        raise ConfigValidationError("slicer.max_tokens must be positive")

    if section["soft_boundary_max_shift"] < 0:
        raise ConfigValidationError("slicer.soft_boundary_max_shift must be non-negative")

    # Проверяем tokenizer
    if section["tokenizer"] != "o200k_base":
        raise ConfigValidationError("slicer.tokenizer must be 'o200k_base'")

    # Проверяем allowed_extensions
    if not section["allowed_extensions"]:
        raise ConfigValidationError("slicer.allowed_extensions cannot be empty")


def _validate_itext2kg_concepts_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [itext2kg_concepts]."""
    required_fields = {
        "model": str,
        "tpm_limit": int,
        "max_completion": int,
        "log_level": str,
        "api_key": str,
        "timeout": int,
        "max_retries": int,
    }

    _validate_required_fields(section, required_fields, "itext2kg_concepts")

    # Проверяем диапазоны
    if section["tpm_limit"] <= 0:
        raise ConfigValidationError("itext2kg_concepts.tpm_limit must be positive")

    if not (1 <= section["max_completion"] <= 100000):
        raise ConfigValidationError("itext2kg_concepts.max_completion must be between 1 and 100000")

    if section["log_level"] not in ["debug", "info", "warning", "error"]:
        raise ConfigValidationError(
            "itext2kg_concepts.log_level must be one of: debug, info, warning, error"
        )

    # Updated api_key check (provider-aware)
    if _placeholder_key(section["api_key"]):
        provider = _provider_of(section)
        env_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        if not os.getenv(env_var):
            raise ConfigValidationError(
                f"itext2kg_concepts.api_key not configured (provider={provider}). Either:\n"
                f"1. Set {env_var} environment variable\n"
                "2. Provide valid key in config.toml"
            )

    if section["timeout"] <= 0:
        raise ConfigValidationError("itext2kg_concepts.timeout must be positive")

    if section["max_retries"] < 0:
        raise ConfigValidationError("itext2kg_concepts.max_retries must be non-negative")

    # Проверяем температуру, если она указана
    if "temperature" in section:
        temp = section["temperature"]
        if not (0 <= temp <= 2):
            raise ConfigValidationError("itext2kg_concepts.temperature must be between 0 and 2")

    # Validate optional response_chain_depth
    if "response_chain_depth" in section:
        depth = section["response_chain_depth"]
        if not isinstance(depth, int) or depth < 0:
            raise ConfigValidationError(
                "itext2kg_concepts.response_chain_depth must be a non-negative integer"
            )

    # Validate optional truncation
    if "truncation" in section:
        truncation = section["truncation"]
        if truncation not in ["auto", "disabled"]:
            raise ConfigValidationError("itext2kg_concepts.truncation must be 'auto' or 'disabled'")


def _validate_itext2kg_graph_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [itext2kg_graph]."""
    required_fields = {
        "model": str,
        "tpm_limit": int,
        "max_completion": int,
        "log_level": str,
        "api_key": str,
        "timeout": int,
        "max_retries": int,
    }

    _validate_required_fields(section, required_fields, "itext2kg_graph")

    # Проверяем диапазоны
    if section["tpm_limit"] <= 0:
        raise ConfigValidationError("itext2kg_graph.tpm_limit must be positive")

    if not (1 <= section["max_completion"] <= 100000):
        raise ConfigValidationError("itext2kg_graph.max_completion must be between 1 and 100000")

    if section["log_level"] not in ["debug", "info", "warning", "error"]:
        raise ConfigValidationError(
            "itext2kg_graph.log_level must be one of: debug, info, warning, error"
        )

    # Updated api_key check (provider-aware)
    if _placeholder_key(section["api_key"]):
        provider = _provider_of(section)
        env_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        if not os.getenv(env_var):
            raise ConfigValidationError(
                f"itext2kg_graph.api_key not configured (provider={provider}). Either:\n"
                f"1. Set {env_var} environment variable\n"
                "2. Provide valid key in config.toml"
            )

    if section["timeout"] <= 0:
        raise ConfigValidationError("itext2kg_graph.timeout must be positive")

    if section["max_retries"] < 0:
        raise ConfigValidationError("itext2kg_graph.max_retries must be non-negative")

    # Проверяем температуру, если она указана
    if "temperature" in section:
        temp = section["temperature"]
        if not (0 <= temp <= 2):
            raise ConfigValidationError("itext2kg_graph.temperature must be between 0 and 2")

    # Validate optional response_chain_depth
    if "response_chain_depth" in section:
        depth = section["response_chain_depth"]
        if not isinstance(depth, int) or depth < 0:
            raise ConfigValidationError(
                "itext2kg_graph.response_chain_depth must be a non-negative integer"
            )

    # Validate optional truncation
    if "truncation" in section:
        truncation = section["truncation"]
        if truncation not in ["auto", "disabled"]:
            raise ConfigValidationError("itext2kg_graph.truncation must be 'auto' or 'disabled'")

    # Validate optional auto_mentions_weight (only for graph)
    if "auto_mentions_weight" in section:
        weight = section["auto_mentions_weight"]
        if not isinstance(weight, (int, float)) or not (0.0 <= weight <= 1.0):
            raise ConfigValidationError(
                "itext2kg_graph.auto_mentions_weight must be between 0.0 and 1.0"
            )


def _validate_dedup_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [dedup]."""
    required_fields = {
        "embedding_model": str,
        "sim_threshold": float,
        "len_ratio_min": float,
        "faiss_M": int,
        "faiss_efC": int,
        "faiss_metric": str,
        "k_neighbors": int,
    }

    _validate_required_fields(section, required_fields, "dedup")

    # Проверяем диапазоны
    if not (0.0 <= section["sim_threshold"] <= 1.0):
        raise ConfigValidationError("dedup.sim_threshold must be between 0.0 and 1.0")

    if not (0.0 <= section["len_ratio_min"] <= 1.0):
        raise ConfigValidationError("dedup.len_ratio_min must be between 0.0 and 1.0")

    if section["faiss_M"] <= 0:
        raise ConfigValidationError("dedup.faiss_M must be positive")

    if section["faiss_efC"] <= 0:
        raise ConfigValidationError("dedup.faiss_efC must be positive")

    if section["faiss_metric"] not in ["INNER_PRODUCT", "L2"]:
        raise ConfigValidationError("dedup.faiss_metric must be 'INNER_PRODUCT' or 'L2'")

    if section["k_neighbors"] <= 0:
        raise ConfigValidationError("dedup.k_neighbors must be positive")

    # Обновленная проверка embedding_api_key (опциональный параметр)
    # Если ключ не задан или placeholder - проверяем env переменные
    embedding_key = section.get("embedding_api_key", "")
    if not embedding_key or embedding_key.startswith("sk-..."):
        # Проверяем сначала OPENAI_EMBEDDING_API_KEY, потом OPENAI_API_KEY
        if not (os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")):
            raise ConfigValidationError(
                "dedup.embedding_api_key not configured. Either:\n"
                "1. Set OPENAI_EMBEDDING_API_KEY or OPENAI_API_KEY environment variable\n"
                "2. Provide valid key in config.toml"
            )


def _validate_refiner_section(section: Dict[str, Any]) -> None:
    """Валидирует секцию [refiner]."""
    required_fields = {
        "run": bool,
        "embedding_model": str,
        "sim_threshold": float,
        "max_pairs_per_node": int,
        "model": str,
        "api_key": str,
        "tpm_limit": int,
        "max_completion": int,
        "timeout": int,
        "max_retries": int,
        # Веса удалены - теперь они прописаны в промптах fw/bw
        # "weight_low": float,
        # "weight_mid": float,
        # "weight_high": float,
    }

    _validate_required_fields(section, required_fields, "refiner")

    # Проверяем диапазоны
    if not (0.0 <= section["sim_threshold"] <= 1.0):
        raise ConfigValidationError("refiner.sim_threshold must be between 0.0 and 1.0")

    if section["max_pairs_per_node"] <= 0:
        raise ConfigValidationError("refiner.max_pairs_per_node must be positive")

    # Updated api_key check (provider-aware)
    if _placeholder_key(section["api_key"]):
        provider = _provider_of(section)
        env_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        if not os.getenv(env_var):
            raise ConfigValidationError(
                f"refiner.api_key not configured (provider={provider}). Either:\n"
                f"1. Set {env_var} environment variable\n"
                "2. Provide valid key in config.toml"
            )

    if section["tpm_limit"] <= 0:
        raise ConfigValidationError("refiner.tpm_limit must be positive")

    if not (1 <= section["max_completion"] <= 100000):
        raise ConfigValidationError("refiner.max_completion must be between 1 and 100000")

    if section["timeout"] <= 0:
        raise ConfigValidationError("refiner.timeout must be positive")

    if section["max_retries"] < 0:
        raise ConfigValidationError("refiner.max_retries must be non-negative")

    # Веса больше не проверяем - они теперь в промптах
    # weights = [section["weight_low"], section["weight_mid"], section["weight_high"]]
    # ...проверки весов удалены...

    # Validate optional response_chain_depth
    if "response_chain_depth" in section:
        depth = section["response_chain_depth"]
        if not isinstance(depth, int) or depth < 0:
            raise ConfigValidationError(
                "refiner.response_chain_depth must be a non-negative integer"
            )

    # Validate optional truncation
    if "truncation" in section:
        truncation = section["truncation"]
        if truncation not in ["auto", "disabled"]:
            raise ConfigValidationError("refiner.truncation must be 'auto' or 'disabled'")


def _validate_required_fields(
    section: Dict[str, Any], required_fields: Dict[str, type], section_name: str
) -> None:
    """Проверяет наличие и типы обязательных полей в секции."""
    for field_name, expected_type in required_fields.items():
        if field_name not in section:
            raise ConfigValidationError(f"Missing required field: {section_name}.{field_name}")

        actual_value = section[field_name]
        if not isinstance(actual_value, expected_type):
            raise ConfigValidationError(
                f"Field {section_name}.{field_name} must be {expected_type.__name__}, "
                f"got {type(actual_value).__name__}"
            )
