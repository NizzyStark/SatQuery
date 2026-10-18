# SatQuery LoRA Adapter for Remote-Sensing VLM

- **Base Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Method**: QLoRA (r=16, alpha=32, dropout=0.05)
- **Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Dataset**: BigEarthNet.txt remote-sensing chat dataset (50 samples)
- **Precision**: `float32`
- **Image Resolution**: Capped to 200704 px
