@@
-    raw_sd = _load_raw_sd(path)
-    up_sd = _extract_upscaler_sd(raw_sd)
-    cfg = _detect_arch(up_sd)
-
-    model = LatentResizer3D(
-        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
-        channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
-        temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
-    )
-    model.load_state_dict(up_sd, strict=True)
+    raw_sd = _load_raw_sd(path)
+    up_sd = _extract_upscaler_sd(raw_sd)
+
+    # Accept checkpoints that were saved with a VideoLatentResizer wrapper
+    # (which prefix parameter names with 'resizer.'). Normalize keys so
+    # they match LatentResizer3D's names (strip 'resizer.' when present).
+    normalized_sd = {}
+    for k, v in up_sd.items():
+        if k.startswith("resizer."):
+            normalized_sd[k[len("resizer."):]] = v
+        else:
+            normalized_sd[k] = v
+    up_sd = normalized_sd
+
+    cfg = _detect_arch(up_sd)
+
+    model = LatentResizer3D(
+        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
+        channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
+        temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
+    )
+
+    # Load non-strictly and report missing/unexpected keys for debugging
+    missing, unexpected = model.load_state_dict(up_sd, strict=False)
+    if missing:
+        print(f"[MinimaxH3-3D] Missing keys when loading state_dict: {missing[:10]} ...")
+    if unexpected:
+        print(f"[MinimaxH3-3D] Unexpected keys in state_dict: {unexpected[:10]} ...")
@@
-    MODEL_CACHE[cache_key] = model
-    print(f"[MinimaxH3-3D] Loaded upscale model: {name}")
-    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
-          f"Attn: forced off | Temporal: {'on' if cfg['temporal_every'] > 0 else 'off'} "
-          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']}) | "
-          f"Backend: {backend_lbl} | Precision: {precision}")
-    return model
+    MODEL_CACHE[cache_key] = model
+    print(f"[MinimaxH3-3D] Loaded upscale model: {name}")
+    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
+          f"Attn: forced off | Temporal: {'on' if cfg['temporal_every'] > 0 else 'off'} "
+          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']}) | "
+          f"Backend: {backend_lbl} | Precision: {precision}")
+    return model
