import nncf
from openvino import Core

# Load model
core = Core()
model = core.read_model("models/dialogpt-openvino")

# Apply weight compression for LLMs
compressed_model = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT4_SYM,  # or INT4_ASYM, INT8
    ratio=0.8,  # Compression ratio
    group_size=128  # Group size for quantization
)

# Save compressed model
import openvino as ov
ov.save_model(compressed_model, "models/dialogpt-compressed.xml")