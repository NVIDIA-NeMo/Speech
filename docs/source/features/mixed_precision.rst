.. _mix_precision:

Mixed Precision Training
------------------------

Mixed precision training significantly enhances computational efficiency by conducting operations in half-precision and fp8 formats, while selectively maintaining minimal data in single-precision to preserve critical information throughout key areas of the network. NeMo now supports FP16, BF16, and FP8 (via Transformer Engine) across most models. Further details will be provided shortly.
