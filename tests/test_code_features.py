from mvrfdet.stage1.code.encoder import StaticCodeEncoder
from mvrfdet.stage1.code.api_features import extract_api_calls


def test_ast_api_code_encoder_shape():
    source = """
import os
import socket

def run(host):
    s = socket.socket()
    os.system('id')
    return s
"""
    calls = extract_api_calls(source)
    assert "socket.socket" in calls
    assert "os.system" in calls
    vector = StaticCodeEncoder().encode([source])
    assert vector.shape == (768,)
