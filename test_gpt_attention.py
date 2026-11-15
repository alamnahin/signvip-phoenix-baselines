import torch

from models.multihead_t2vqpgpt import Text2VQPoseGPT


def make_model(attention):
    torch.manual_seed(7)
    return Text2VQPoseGPT(
        num_vq=12,
        embed_dim=16,
        clip_dim=8,
        block_size=8,
        num_layers=2,
        n_head=2,
        drop_out_rate=0,
        pose_size=2,
        head_layers=1,
        attn_type=attention,
    ).eval()


@torch.no_grad()
def test_cached_logits_match_full_forward():
    clip = torch.randn(1, 8)
    tokens = torch.randint(0, 10, (1, 8))

    for attention in ("global", "local:2"):
        model = make_model(attention)
        model.clear_cache()
        for step in range(4):
            full_input = [] if step == 0 else tokens[:, : step * 2]
            cached_input = [] if step == 0 else tokens[:, (step - 1) * 2 : step * 2]
            full_logits = model(full_input, clip, flat=False, use_cache=False)[:, -1]
            cached_logits = model(
                cached_input,
                clip,
                flat=False,
                use_cache=True,
                shift_t=step,
            )[:, -1]
            torch.testing.assert_close(full_logits, cached_logits, atol=1e-5, rtol=1e-5)


def test_invalid_attention_mode_is_rejected():
    try:
        make_model("local")
    except ValueError as error:
        assert "Unsupported attention type" in str(error)
    else:
        raise AssertionError("Invalid attention mode was accepted")


if __name__ == "__main__":
    test_cached_logits_match_full_forward()
    test_invalid_attention_mode_is_rejected()
    print("GPT attention checks passed.")
