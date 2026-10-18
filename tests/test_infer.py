from pathlib import Path

from indentra import infer

BASE_COMMAND = ["llama-server", "-m", "m.gguf", "--port", "8091", "-c", "4096", "-np", "1", "-ngl", "99", "--no-webui", "--seed", "0"]


def test_llama_server_command_is_unchanged_without_lora():
    assert infer.LlamaServer(Path("m.gguf"))._command == BASE_COMMAND


def test_llama_server_loads_a_lora_adapter_when_given():
    assert infer.LlamaServer(Path("m.gguf"), lora=Path("adapters/x/lora.gguf"))._command == BASE_COMMAND + ["--lora", "adapters/x/lora.gguf"]
