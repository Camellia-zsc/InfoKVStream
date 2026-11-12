"""Read-only post-RoPE Q capture; native FA2/SDPA still computes attention."""
import copy

def _interface(module, query, key, value, attention_mask, *, scaling, dropout=0.0, **kwargs):
    if module._eventkv_query_capture is not None:
        module._eventkv_query_capture(int(module.layer_idx), query.detach())
    fn = module._eventkv_eager if module._eventkv_output_attentions else module._eventkv_original
    return fn(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)

class QueryCaptureAdapter:
    def __init__(self, layers, *, query_capture=None):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
        ALL_ATTENTION_FUNCTIONS.register("eventkv_readonly_capture", _interface)
        self._original_configs = []
        self._hook_handles = []
        for layer in layers:
            attn = layer.self_attn
            if hasattr(attn, "_eventkv_original"):
                self.close()
                raise ValueError("attention already has a query capture adapter")
            if getattr(attn, "sliding_window", None) is not None:
                self.close()
                raise NotImplementedError("requires full-attention layers")
            config = attn.config
            key = config._attn_implementation
            private = copy.copy(config)
            private._attn_implementation = "eventkv_readonly_capture"
            self._original_configs.append((attn, config))
            attn._eventkv_original = eager_attention_forward if key == "eager" else ALL_ATTENTION_FUNCTIONS[key]
            attn._eventkv_eager = eager_attention_forward
            attn._eventkv_query_capture = query_capture
            attn._eventkv_output_attentions = False
            attn.config = private
            self._hook_handles.append(attn.register_forward_pre_hook(self._flags, with_kwargs=True))

    @staticmethod
    def _flags(module, args, kwargs):
        module._eventkv_output_attentions = bool(kwargs.get("output_attentions", False))

    def close(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        for attn, config in self._original_configs:
            attn.config = config
            for key in ("_eventkv_original", "_eventkv_eager", "_eventkv_query_capture", "_eventkv_output_attentions"):
                delattr(attn, key)
        self._original_configs = []
