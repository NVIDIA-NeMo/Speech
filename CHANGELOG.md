# Changelog

## NVIDIA Neural Modules 2.0.0rc1

### Highlights

#### Large language models

- PEFT: QLoRA support, LoRA/QLora for Mixture-of-Experts (MoE) dense layer
- State Space Models & Hybrid Architecture support (Mamba2 and NV-Mamba2-hybrid)
- Support Nemotron, Minitron, Gemma2, Qwen, RAG
- Custom Tokenizer training in NeMo
- Update the Auto-Configurator for EP, CP and FSDP

#### Multimodal

- NeVA: Add SOTA LLM backbone support (Mixtral/LLaMA3) and suite of model parallelism support (PP/EP)
- Support Language Instructed Temporal-Localization Assistant (LITA) on top of video NeVA

#### ASR

- SpeechLM and SALM
- Adapters for Canary Customization
- Pytorch allocator in PyTorch 2.2 improves training speed up to 30% for all ASR models
- Cuda Graphs for Transducer Inference
- Replaced webdataset with Lhotse - gives up to 2x speedup
- Transcription Improvements - Speedup and QoL Changes
- ASR Prompt Formatter for multimodal Canary

#### Export & Deploy

- In framework PyTriton deployment with backends: - PyTorch - vLLM - TRT-LLM update to 0.10
- TRT-LLM C++ runtime

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- Support dataloader as input to `audio` for transcription by @titu1994 :: PR: #9201  
- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205  
- Fix Online_Offline_Microphone_VAD_Demo.ipynb by @stevehuang52 :: PR: #9251  
- Remove .nemo instead of renaming by @mikolajblaz :: PR: #9281  
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. by @galv :: PR: #9347
- Revert "Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer." by @titu1994 :: PR: #9351
- Prompt formatter API and canary transcribe tensor input support by @pzelasko :: PR: #9206
- Fix prompt formatter's defaults=None case in multi-task model by @pzelasko :: PR: #9366
- move AED chunked infer script by @stevehuang52 :: PR: #9367
- Use model-cast-to-bfloat16 rather than AMP-to-bfloat16 for inference. by @galv :: PR: #9198
- ci: Fix `L2_Segmentation_Tool_Parallel_ctc_segmentation_test_L2_Eng_C… by @ko3n1g :: PR: #9399
- Fix logging message for ASR by @titu1994 :: PR: #9469
- Add support to change Multi task model prompt by @titu1994 :: PR: #9542
- Enable encoder adapters for Canary and MultiTaskAED models by @titu1994 :: PR: #9409
- Audio model collection by @anteju :: PR: #9263
- TitaNet Batch Verify Speaker by @monica-sekoyan :: PR: #9337
- Fix the arguments  of forward_for_export function in msdd_models by @tango4j :: PR: #9624
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- refactor: notebook branch release by @ko3n1g :: PR: #9711
- Canary Adapters tutorial (#9670) by @nithinraok :: PR: #9777
- typos and branch name update to r2.0.0rc1 by @nithinraok :: PR: #9846
- Fix RNNT alignments test by @artbataev :: PR: #9770
- By default trust remote code from HF Datasets by @nithinraok :: PR: #9886
- Temporarily disable cuda graph based RNN-T greedy inference for r2.0.0rc1 by @galv :: PR: #9904
- Enable CUDA graphs by default, but require CUDA 12.6 for full graphs by @artbataev :: PR: #9919
- update branch name for script by @nithinraok :: PR: #9936
- updte branch by @nithinraok :: PR: #9942
</details>

#### TTS

<details><summary>Changelog</summary>

- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205
- Add mel codec checkpoints by @anteju :: PR: #9228
- GPU unit tests: Mark flaky tests to be fixed by @pablo-garay :: PR: #9559
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- refactor: notebook branch release by @ko3n1g :: PR: #9711

</details>

#### LLM/Multimodal
  
<details><summary>Changelog</summary>

- Update nemo.export module for quantized models by @janekl :: PR: #9218
- Add save option to the TRT-LLM export test script by @oyilmaz-nvidia :: PR: #9221
- Checkpoint resuming compatible for 2403 container by @suiyoubi :: PR: #9199
- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205
- use get with fallback when reading checkpoint_callback_params by @akoumpa :: PR: #9223
- Revert rope fusion defaults by @cuichenx :: PR: #9237
- fix import by @akoumpa :: PR: #9240
- Add TRT-LLM params like max_num_tokens and opt_num_tokens by @oyilmaz-nvidia :: PR: #9210
- sum-reduce grad_norm in DP+CP domain by @erhoo82 :: PR: #9262
- Alit/bert convert fix by @JRD971000 :: PR: #9285
- conv1d stable version by @JRD971000 :: PR: #9330
- Fix trainer builder when exp_manager is not in config by @yaoyu-33 :: PR: #9293
- Fix Peft Weights Loading in NeVA by @yaoyu-33 :: PR: #9341
- Skip sequence_parallel allreduce when using Mcore DistOpt by @akoumpa :: PR: #9344
- Fix FSDP gradient calculation with orig params by @janEbert :: PR: #9335
- TRT-LLM Export Code Cleanup by @oyilmaz-nvidia :: PR: #9270
- support null/None truncation field by @arendu :: PR: #9355
- NeVa token fusion by @paul-gibbons :: PR: #9245
- bugfix if using mcore distOpt with sft by @akoumpa :: PR: #9356
- Re-org export code by @oyilmaz-nvidia :: PR: #9353
- QLoRA by @cuichenx :: PR: #9340
- PeFT fix for distOpt by @akoumpa :: PR: #9392
- [NeMo-UX] Integrating mcore's DistributedDataParallel into MegatronStrategy by @marcromeyn :: PR: #9387
- cherry pick of #9266 by @dimapihtar :: PR: #9411
- Enable specifying alpha for PTQ INT8 SmoothQuant method by @janekl :: PR: #9423
- add support for new mcore ds features by @dimapihtar :: PR: #9388
- LoRA for MoE Layer by @cuichenx :: PR: #9396
- Mistral-7B: apply user's precision to output checkpoint by @akoumpa :: PR: #9222
- Add option to merge distributed optimizer buckets by @timmoon10 :: PR: #9414
- TRT-LLM 0.10 Update by @oyilmaz-nvidia :: PR: #9402
- In-framework deployment by @oyilmaz-nvidia :: PR: #9438
- Bugfix missing variables and argument changes to MegatronPretrainingRandomSampler by @jstjohn :: PR: #9458
- Hyena Operator by @guyjacob :: PR: #9264
- Refactor Quantizer for reusing in QAT by @kevalmorabia97 :: PR: #9276
- move load state dict after initialize parallel state in nlp_model by @ryxli :: PR: #9382
- Enable user to optionally upgrade Megatron by @jstjohn :: PR: #9478
- Fix unwrap model by @cuichenx :: PR: #9480
- fix operator precedence by @akoumpa :: PR: #9403
- [NeMo-UX] Adding context- & expert-parallelism to MegatronStrategy by @marcromeyn :: PR: #9525
- update mcoreddp call by @akoumpa :: PR: #9345
- mcore distOpt restore fix by @akoumpa :: PR: #9421
- vLLM Export Support by @apanteleev :: PR: #9381
- PL: Delete precision if using plugin. TODO switch to MegatronTrainerB… by @akoumpa :: PR: #9535
- extend get_gpt_layer_modelopt_spec to support MoE by @akoumpa :: PR: #9532
- fix mock data generation for legacy dataset by @dimapihtar :: PR: #9530
- add reset learning rate functionality by @dimapihtar :: PR: #9372
- Use closed-formula to round by multiple by @akoumpa :: PR: #9307
- GPU unit tests: Mark flaky tests to be fixed by @pablo-garay :: PR: #9559
- Consolidate gpt continue training script into pretraining script by @yaoyu-33 :: PR: #9413
- Enable encoder adapters for Canary and MultiTaskAED models by @titu1994 :: PR: #9409
- PTQ refinements by @janekl :: PR: #9574
- Add ModelOpt QAT example for Llama2 SFT model by @kevalmorabia97 :: PR: #9326
- Multimodal projection layer adapter fix for PP>1 by @paul-gibbons :: PR: #9445
- Add offline quantization script for QLoRA deployment by @cuichenx :: PR: #9455
- Make QLoRA more model-agnostic by @cuichenx :: PR: #9488
- Set n_gpu to None in nemo export by @oyilmaz-nvidia :: PR: #9593
- [NeMo-UX] Fix Megatron-optimizer by @marcromeyn :: PR: #9599
- Chat template support for megatron_gpt_eval.py by @akoumpa :: PR: #9354
- [NeMo-UX] Add PEFT by @cuichenx :: PR: #9490
- Alit/mamba tmp by @JRD971000 :: PR: #9612
- Enable MCore checkpointing optimizations by @mikolajblaz :: PR: #9505
- Change mixtral moe key name for trt-llm by @oyilmaz-nvidia :: PR: #9620
- fix ckpt load bug by @dimapihtar :: PR: #9621
- Alit/mamba by @JRD971000 :: PR: #9575
- Unwrap ckpt_io for model opt (async save) by @mikolajblaz :: PR: #9622
- MCore T5 support for NeMo - Training by @huvunvidia :: PR: #9432
- [Nemo-UX] Expose transformer_layer_spec inside GPTConfig by @marcromeyn :: PR: #9592
- Update NeMo Clip to Use MCore Modules by @yaoyu-33 :: PR: #9594
- Mistral + Mixtral Support for NeVa by @paul-gibbons :: PR: #9459
- Adding support for mcore generate by @shanmugamr1992 :: PR: #9566
- Improve error messaging during trt-llm export by @oyilmaz-nvidia :: PR: #9638
- [Cherrypick] support lora when kv_channel != hidden_size / num_heads by @cuichenx :: PR: #9644
- Parametrize FPS group by @mikolajblaz :: PR: #9648
- Cherry-pick megatron export fix from main by @borisfom :: PR: #9643
- add documentation for reset_lr feature by @dimapihta
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- Cherry pick: LITA Integration by @Slyne :: PR: #9684
- SDXL improvements (and support for Draft+) by @rohitrango :: PR: #9654
- Gemma 2 by @cuichenx :: PR: #9672
- Allows non-strict load with distributed checkpoints by @mikolajblaz :: PR: #9613
- refactor: notebook branch release by @ko3n1g :: PR: #9711
- [NeMo-UX] Make TE and Apex dependencies optional by @ashors1 :: PR: #9550
- Alit/r2.0.0 by @JRD971000 :: PR: #9718
- Manually cherry-pick from PR 9679 (PR to main - Support SFT/Eval/PEFT for mcore T5) by @huvunvidia :: PR: #9737
- In framework export by @oyilmaz-nvidia :: PR: #9658
- T5 changes based on mcore changes by @pablo-garay :: PR: #9829
- [NeMo-UX] Use single instance of loss reductions in GPTModel by @hemildesai :: PR: #9801
- deprecate NeMo NLP tutorial by @dimapihtar :: PR: #9864
- Disable nvFuser setup with PyTorch 23.11 and later by @athitten :: PR: #9837
- make torch_dist ckpt strategy as default by @dimapihtar :: PR: #9852
- add rampup bs documentation by @dimapihtar :: PR: #9884
- copy of #9576 by @dimapihtar :: PR: #9986
- Support Nvidia Torch and Arch versions by @thomasdhc :: PR: #9897
- Bug fix for pooler causing dist checkpointing exception by @shanmugamr1992 :: PR: #10008

</details>

#### Export

<details><summary>Changelog</summary>

- Update nemo.export module for quantized models by @janekl :: PR: #9218
- Add save option to the TRT-LLM export test script by @oyilmaz-nvidia :: PR: #9221
- Add TRT-LLM params like max_num_tokens and opt_num_tokens by @oyilmaz-nvidia :: PR: #9210
- TRT-LLM Export Code Cleanup by @oyilmaz-nvidia :: PR: #9270
- Re-org export code by @oyilmaz-nvidia :: PR: #9353
- Use TensorRT-LLM native parameter names in nemo.export module by @janekl :: PR: #9424
- TRT-LLM 0.10 Update by @oyilmaz-nvidia :: PR: #9402
- vLLM Export Support by @apanteleev :: PR: #9381
- Add page context fmha option in TensorRTLLM export by @meatybobby :: PR: #9526
- Test C++ runtime on demand in nemo_export.py to avoid possible OOMs by @janekl :: PR: #9544
- Fix nemo export test by @oyilmaz-nvidia :: PR: #9547
- Add tps and pps params to the export script by @oyilmaz-nvidia :: PR: #9558
- Add Multimodal Exporter by @meatybobby :: PR: #9256
- Set n_gpu to None in nemo export by @oyilmaz-nvidia :: PR: #9593
- Inflight nemo model export support by @JimmyZhang12 :: PR: #9527
- vLLM Export Improvements by @apanteleev :: PR: #9596
- Akoumparouli/nemo ux mixtral export by @akoumpa :: PR: #9603
- Change mixtral moe key name for trt-llm by @oyilmaz-nvidia :: PR: #9620
- Fix the arguments  of forward_for_export function in msdd_models by @tango4j :: PR: #9624
- Improve error messaging during trt-llm export by @oyilmaz-nvidia :: PR: #9638
- Cherry-pick megatron export fix from main by @borisfom :: PR: #9643
- In framework export by @oyilmaz-nvidia :: PR: #9658
- Add missing imports for torch dist ckpt in export by @oyilmaz-nvidia :: PR: #9826~

</details>




#### Bugfixes
  
<details><summary>Changelog</summary>

- use get with fallback when reading checkpoint_callback_params by @akoumpa :: PR: #9223
- fix import by @akoumpa :: PR: #9240
- Remove .nemo instead of renaming by @mikolajblaz :: PR: #9281
- call set_expert_model_parallel_world_size instead of set_cpu_expert_m… by @akoumpa :: PR: #9275
- Fix typos in Mixtral NeMo->HF and Starcoder2 NeMo->HF conversion scripts by @evellasques :: PR: #9325
- Skip sequence_parallel allreduce when using Mcore DistOpt by @akoumpa :: PR: #9344
- Add OpenAI format response to r2.0.0rc1 by @athitten :: PR: #9796
- [NeMo UX] Support generating datasets using different train/valid/test distributions by @ashors1 :: PR: #9771
- Add missing imports for torch dist ckpt in export by @oyilmaz-nvidia :: PR: #9826

</details>

#### General Improvements

<details><summary>Changelog</summary>

- [Nemo CICD] run_cicd_for_release_branches_also by @pablo-garay :: PR: #9213
- rename paths2audiofiles to audio by @github-actions[bot] :: PR: #9220
- Fix ASR_Context_Biasing.ipynb contains FileNotFoundError by @github-actions[bot] :: PR: #9234
- ci: Remove duplicated job by @ko3n1g :: PR: #9258
- Fix document links by @yaoyu-33 :: PR: #9260
- Pin transformers by @github-actions[bot] :: PR: #9273
- Fix loading github raw images on notebook by @github-actions[bot] :: PR: #9283
- Accept None as an argument to decoder_lengths in GreedyBatchedCTCInfer::forward by @github-actions[bot] :: PR: #9278
- Refactor Sequence Packing Script by @cuichenx :: PR: #9271
- [Nemo-UX] Move code to collections + fix some small bugs by @marcromeyn :: PR: #9277
- Fix typo in HF tutorial by @github-actions[bot] :: PR: #9304
- Expand documentation for data parallelism and distributed optimizer by @timmoon10 :: PR: #9227
- Install alerting by @ko3n1g :: PR: #9311
- typos by @github-actions[bot] :: PR: #9315
- FP8 feature documentation by @ksivaman :: PR: #9265
- [Nemo CICD] Comment out flaky tests by @pablo-garay :: PR: #9333
- Fixed typos in README.rst by @gdevakumar :: PR: #9322
- Update README.rst to clarify installation via Conda by @SimonCW :: PR: #9323
- [Nemo CICD] update flaky test by @pablo-garay :: PR: #9339
- fix lora and ptuning and isort/black by @github-actions[bot] :: PR: #9295
- Fix P-tuning for Llama based models by @github-actions[bot] :: PR: #9300
- add large model stable training fix and contrastive loss update for variable seq by @github-actions[bot] :: PR: #9348
- Guard cuda memory allocator update by @github-actions[bot] :: PR: #9313
- [Nemo CICD] Remove unnecessary commented out code by @pablo-garay :: PR: #9364
- Update Gemma conversion script by @yaoyu-33 :: PR: #9365
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. (#9347) by @github-actions[bot] :: PR: #9371
- Re-enable cuda graphs in training modes. by @github-actions[bot] :: PR: #9343
- fix typo infer_seq_lenght -> infer_seq_length by @akoumpa :: PR: #9370
- Make a backward compatibility for old MSDD configs in label models by @github-actions[bot] :: PR: #9378
- Dgalvez/fix greedy batch strategy name r2.0.0rc0 by @github-actions[bot] :: PR: #9253
- Update README.rst by @jgerh :: PR: #9393
- Force diarizer to use CUDA if cuda is available and if device=None. by @github-actions[bot] :: PR: #9390
- ci: Properly catch failed tests by introduction of workflow templates by @ko3n1g :: PR: #9324
- Fix T5 G2P Input and Output Types by @github-actions[bot] :: PR: #9269
- Huvu/rag pipeline citest by @huvunvidia :: PR: #9384
- Fix circular import for MM dataprep notebook by @github-actions[bot] :: PR: #9292
- add check if num layers is divisible by pp size by @github-actions[bot] :: PR: #9298
- [Nemo CICD] timeouts fix by @pablo-garay :: PR: #9407
- [NeMo-UX] Removing un-used ModelConfig class by @marcromeyn :: PR: #9389
- Add tutorial for Llama-3-8B lora training and deployment by @shashank3959 :: PR: #9359
- [NeMo-UX] Removing default_path from ModelConnector by @marcromeyn :: PR: #9401
- Fix README by @ericharper :: PR: #9415
- [SD] Fix SD CUDA Graph Failure by @alpha0422 :: PR: #9319
- [NeMo-UX] Adding file-lock to Connector by @marcromeyn :: PR: #9400
- Add Dev Container Bug Report by @pablo-garay :: PR: #9430
- Akoumparouli/profiling docs by @akoumpa :: PR: #9420
- ci: Enrich notifications by @ko3n1g :: PR: #9412
- Fix failing RIR unit test with lhotse 1.24+ by @pzelasko :: PR: #9444
- [NeMo-UX] Adding support for mcore distributed optimizer by @marcromeyn :: PR: #9435
- Use ModelOpt build_tensorrt_llm for building engines for qnemo checkpoints by @janekl :: PR: #9452
- ci(notifications): Fix extraction of last 2K chars by @ko3n1g :: PR: #9450
- Update readme with mlperf news by @ericharper :: PR: #9457
- [NeMo-UX] Add nsys callback by @ashors1 :: PR: #9461
- [NeMo UX] Introducing optimizer module by @marcromeyn :: PR: #9454
- Fix minor import bug in deploy module by @oyilmaz-nvidia :: PR: #9463
- ci(notifications): Fetch all jobs by @ko3n1g :: PR: #9465
- Update build_dataset.py by @stevehuang52 :: PR: #9467
- bionemo: bn2/add pipelineparallel dtype by @skothenhill-nv :: PR: #9475
- [NeMo-UX] Integrate experiment manager features with NeMo-UX APIs by @ashors1 :: PR: #9460
- Add python_requires by @galv :: PR: #9431
- [NeMo-UX] Fixing imports of NeMoLogging, AutoResume & ModelCheckpoint by @marcromeyn :: PR: #9476
- Modelopt Refactor for SDXL Quantization by @suiyoubi :: PR: #9279
- [NeMo-UX] Fixing defaults in llm.train & Mistral7BModel by @marcromeyn :: PR: #9486
- In framework deploy using deploy script by @oyilmaz-nvidia :: PR: #9468
- [NeMo-UX] Integrate tokenizer import into model.import_ckpt by @marcromeyn :: PR: #9485
- append to file by @malay-nagda :: PR: #9483
- [NeMo-UX] Fix bug in import_ckpt by @marcromeyn :: PR: #9492
- Add nemotron news by @ericharper :: PR: #9510
- Add CICD test for Stable Diffusion by @michal2409 :: PR: #9464
- Akoumparouli/nemo ux mixtral by @akoumpa :: PR: #9446
- [NeMo-UX] Llama and Gemma by @cuichenx :: PR: #9528
- [NeMo-UX] minor logging bug fixes by @ashors1 :: PR: #9529
- Update neva conversion script from and to HF by @yaoyu-33 :: PR: #9296
- [Nemo-UX] IO fixes by @marcromeyn :: PR: #9512
- Fix lhotse tests for v1.24.2 by @pzelasko :: PR: #9546
- [Nemo CICD] Make GPU Unit Tests non-optional by @pablo-garay :: PR: #9551
- Add Python AIStore SDK to container and bump min Lhotse version by @pzelasko :: PR: #9537
- [NeMo-UX] Fix tokenizer IO by @marcromeyn :: PR: #9555
- [NeMo UX] Move mistral_7b.py to mistral.py by @akoumpa :: PR: #9545
- ci: Do not attempt to send slack on fork by @ko3n1g :: PR: #9556
- Fix SDXL incorrect name in Docs by @suiyoubi :: PR: #9534
- Bump PTL version by @athitten :: PR: #9557
- [Resiliency] Straggler detection by @jbieniusiewi :: PR: #9473
- [NeMo-UX] Switch to torch_dist as default distributed checkpointing backend by @ashors1 :: PR: #9541
- [NeMo-UX] Checkpointing bug fixes by @ashors1 :: PR: #9562
- Expose MCore path_to_cache option by @maanug-nv :: PR: #9570
- [NeMo-UX] Fix Trainer serialization by @marcromeyn :: PR: #9571
- Update click version requirement by @thomasdhc :: PR: #9580
- [Fault tolerance] Heartbeat detection by @maanug-nv :: PR: #9352
- [Nemo-UX] Add fabric-API for manual forward-pass by @marcromeyn :: PR: #9577
- [Nemo-UX] Add SDK-factories to llm-collection by @marcromeyn :: PR: #9589
- [NeMo-UX] Some improvements to NeMoLogger by @marcromeyn :: PR: #9591
- Set no_sync_func & grad_sync_fucn by @akoumpa :: PR: #9601
- [NeMo-UX] Fix nemo logger when trainer has no loggers by @ashors1 :: PR: #9607
- Fix the dictionary  format returned by the `scheduler` method by @sararb :: PR: #9609
- [NeMo-UX] Dataloading enhancements and bug fixes by @ashors1 :: PR: #9595
- Fix serialization of AutoResume by @sararb :: PR: #9616
- Jsonl support by @adityavavre :: PR: #9611
- Akoumparouli/mistral import instruct chat template fix by @akoumpa :: PR: #9567
- Remove .cuda calls, use device isntead by @akoumpa :: PR: #9602
- fix converter defautl args by @akoumpa :: PR: #9565
- fix: remove non_blocking from PTL's .cuda call by @akoumpa :: PR: #9618
- NeVA Minor Fixes by @yaoyu-33 :: PR: #9608
- [NeMo-UX] fix pretrianing data sizes and weights by @cuichenx :: PR: #9627
- [NeMo-UX] async checkpointing support by @ashors1 :: PR: #9466
- Change default parallel_save to False by @mikolajblaz :: PR: #9632
- Add REST API to deploy module by @athitten :: PR: #9539
- ci: Timeout per step, not job by @ko3n1g :: PR: #9635
- [NeMo-UX] Fix when optimizers are setup for PEFT by @marcromeyn :: PR: #9619
- [NeMo-UX] Fix pipeline parallel bug by @ashors1 :: PR: #9637
- Fixing import error fior llama-index (RAG pipeline) by @pablo-garay :: PR: #9662
- llama CI fix by @rohitrango :: PR: #9663
- [NeMo-UX] Make 'load_directly_on_device' configurable by @ashors1 :: PR: #9657
- [Nemo-UX] Including all trainable-params in a PEFT-checkpoint by @marcromeyn :: PR: #9650
- [NeMo-UX] Fix imports so local configuration of runs works again by @marcromeyn :: PR: #9690
- Set TE flag in legacy -> mcore conversion script by @terrykong :: PR: #9722
- Update starthere docs text by @erastorgueva-nv :: PR: #9724
- TorchAudio installation workaround for incorrect `PYTORCH_VERSION` variable by @artbataev :: PR: #9736
- [NeMo-UX] Match nemo 1's default behavior for drop_last and pad_samples_to_global_batch_size by @ashors1 :: PR: #9707
- add a bit more for timeout (#9702) by @pablo-garay :: PR: #9754
- Fix missing parallelisms by @maanug-nv :: PR: #9725
- update branch by @nithinraok :: PR: #9764
- Fix data preprocessing script by @cuichenx :: PR: #9759
- vLLM 0.5.1 update by @apanteleev :: PR: #9779
- upper bound hf-hub by @akoumpa :: PR: #9805
- Fix few issues and docs for neva and clip in r2.0.0rc1 by @yaoyu-33 :: PR: #9681
- add dummy vision and text transformer config (assumed mcore to be false) by @rohitrango :: PR: #9699
- fix lita bugs by @Slyne :: PR: #9810
- [NeMo-UX] Log `val_loss` by @ashors1 :: PR: #9814
- [NeMo-UX] Fix some dataloading bugs by @ashors1 :: PR: #9807
- [NeMo-UX] Adding recipes by @marcromeyn :: PR: #9720
- [NeMo-UX] Set async_save from strategy rather than ModelCheckpoint by @ashors1 :: PR: #9800
- Fix hf hub for 0.24+ by @titu1994 :: PR: #9806
- [NeMo-UX] Fix a minor bug with async checkpointing by @ashors1 :: PR: #9856
- [NeMo-UX] make progress bar easier to parse by @ashors1 :: PR: #9877
- Docs: add "Nemo Fundamentals" page by @erastorgueva-nv :: PR: #9835
- Create __init__.py by @stevehuang52 :: PR: #9892
- [NeMo-UX] Fixes to make PreemptionCallback work by @hemildesai :: PR: #9830
- Fix Docker build. Make Dockerfile consistent with CI by @artbataev :: PR: #9784
- Multimodal data prep notebook fix by @cuichenx :: PR: #9910
- [NeMo-UX] Add distributed checkpointing unit tests by @ashors1 :: PR: #9794
- r2.0.0rc1 fix for dist checkpoint loading by @yaoyu-33 :: PR: #9854
- [NeMo-UX] Rename sdk references to NeMo Run by @hemildesai :: PR: #9872
- [NeMo-UX] Fix some serialization bugs by @ashors1 :: PR: #9868
- add mixtral neva tutorial (moe + token fusion + siglip) by @paul-gibbons :: PR: #9926
- [NeMo-UX] Add more NeMo Logger tests by @ashors1 :: PR: #9795
- Akoumparouli/mixtral fixes for r2.0.0rc1 by @akoumpa :: PR: #9911
- R2.0.0rc1 clip fix by @Slyne :: PR: #9871
- [NeMo-UX] Add missing docstrings and update some defaults by @ashors1 :: PR: #9895
- Add REST service requirements.txt by @oyilmaz-nvidia :: PR: #9923
- add bert latest fix by @JRD971000 :: PR: #9921
- remove empy reconfigure_limit_batches by @akoumpa :: PR: #9934
- fix mem by @terrykong :: PR: #9964
- Run a sample query for a quantized model conditionally by @janekl :: PR: #9965
- Add pydantic-settings  by @oyilmaz-nvidia :: PR: #9961
- Resiliency features update by @jbieniusiewi :: PR: #9714
- [NeMo-UX] Wrap task config save in a try/except by @ashors1 :: PR: #9956
- [NeMo-UX] Update default PTL logging `save_dir` by @ashors1 :: PR: #9954
- Fix lita tutorial by @Slyne :: PR: #9980
- Add deploy and REST API support to NeMo 2.0 by @athitten :: PR: #9834
- ci: Allow changelog manual (#10156) by @ko3n1g :: PR: #10157
- docs: Add changelog by @ko3n1g :: PR: #10155
- add manifest file by @ko3n1g :: PR: #10161

</details>

## NVIDIA Neural Modules 2.0.0rc0

### Highlights

#### LLM and MM

##### Models

- Megatron Core RETRO
  - Pre-training
  - Zero-shot Evaluation

- Pretraining, conversion, evaluation, SFT, and PEFT for:
  - Mixtral 8X22B
  - Llama 3
  - SpaceGemma

- Embedding Models Fine Tuning
  - Mistral
  - BERT

- BERT models
  - Context Parallel
  - Distributed checkpoint

- Video capabilities with NeVa

##### Performance

- Distributed Checkpointing
  - Torch native backend
  - Parallel read/write
  - Async write

- Multimodal LLM (LLAVA/NeVA)
  - Pipeline Parallelism support
  - Sequence packing support

##### Export

- Integration of Export & Deploy Modules into NeMo Framework container
  - Upgrade to TRT-LLM 0.9

#### Speech (ASR & TTS)

##### Models

- AED Multi Task Models (Canary) - Multi-Task Multi-Lingual Speech Recognition / Speech Translation model
- Multimodal Domain - Speech LLM supporting SALM Model
- Parakeet-tdt_ctc-1.1b Model - RTFx of > 1500 (can transcribe 1500 seconds of audio in 1 second)
- Audio Codec 16kHz Small - NeMo Neural Audio Codec for discretizing speech for use in LLMs
  - mel_codec_22khz_medium
  - mel_codec_44khz_medium

##### Perf Improvements

- Transcribe() upgrade - Enables one line transcribe with files, tensors, data loaders
- Frame looping algorithm for RNNT faster decoding - Improves Real Time Factor (RTF) by 2-3x
- Cuda Graphs + Label-Looping algorithm for RNN-T and TDT Decoding - Transducer Greedy decoding at over 1500x RTFx, on par with CTC Non-Autoregressive models
- Semi Sorted Batching support - External User contribution that speeds up training by 15-30%.

##### Customization

- Context biasing for CTC word stamping - Improve accuracy for custom vocabulary and pronunciation
  - Longform Inference
  - Longform inference support for AED models
- Transcription of multi-channel audio for AED models

##### Misc

- Upgraded webdataset - Speech and LLM / Multimodal unified container

### Detailed Changelogs

#### ASR
  
<details><summary>Changelog</summary>

- Enable using hybrid asr models in CTC Segmentation tool by @erastorgueva-nv :: PR: #8828
- TDT confidence fix by @GNroy :: PR: #8982
- Fix union type annotations for autodoc+mock-import rendering by @pzelasko :: PR: #8956
- NeMo dev doc restructure by @yaoyu-33 :: PR: #8896
- Improved random seed configuration for Lhotse dataloaders with docs by @pzelasko :: PR: #9001
- Fix #8948, allow preprocessor to be stream captured to a cuda graph when doing per_feature normalization by @galv :: PR: #8964
- [ASR] Support for transcription of multi-channel audio for AED models by @anteju :: PR: #9007
- Add ASR latest news by @titu1994 :: PR: #9073
- Fix docs errors and most warnings by @erastorgueva-nv :: PR: #9006
- PyTorch CUDA allocator optimization for dynamic batch shape dataloading in ASR by @pzelasko :: PR: #9061
- RNN-T and TDT inference: use CUDA graphs by default by @artbataev :: PR: #8972
- Fix #8891 by supported GPU-side batched CTC Greedy Decoding by @galv :: PR: #9100
- Update branch for notebooks and ci in release by @ericharper :: PR: #9189
- Enable CUDA graphs by default only for transcription by @artbataev :: PR: #9196
- rename paths2audiofiles to audio by @nithinraok :: PR: #9209
- Fix ASR_Context_Biasing.ipynb contains FileNotFoundError by @andrusenkoau :: PR: #9233
- Cherrypick: Support dataloader as input to `audio` for transcription (#9201) by @titu1994 :: PR: #9235
- Update Online_Offline_Microphone_VAD_Demo.ipynb by @stevehuang52 :: PR: #9252
- Dgalvez/fix greedy batch strategy name r2.0.0rc0 by @galv :: PR: #9243
- Accept None as an argument to decoder_lengths in GreedyBatchedCTCInfer::forward by @galv :: PR: #9246
- Fix loading github raw images on notebook by @nithinraok :: PR: #9282
- typos by @nithinraok :: PR: #9314
- Re-enable cuda graphs in training modes. by @galv :: PR: #9338
- add large model stable training fix and contrastive loss update for variable seq by @nithinraok :: PR: #9259
- Fix conv1d package in r2.0.0rc0  by @pablo-garay :: PR: #9369
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. (#9347) by @titu1994 :: PR: #9350
- Make a backward compatibility for old MSDD configs in label models by @tango4j :: PR: #9377
- Force diarizer to use CUDA if cuda is available and if device=None. by @tango4j :: PR: #9380

</details>
  
#### TTS

<details><summary>Changelog</summary>

- [TTS] Add tutorial for training audio codecs by @rlangman :: PR: #8723
- Update radtts.py by @blisc :: PR: #9097
- [Nemo CICD] RADTTS test optional by @pablo-garay :: PR: #9112
- Remove Radtts CI test by @blisc :: PR: #9144
- Fix T5 G2P Input and Output Types by @blisc :: PR: #9224

</details>

#### LLM and MM

<details><summary>Changelog</summary>

- Rachitg/dpa by @rachitgarg91 :: PR: #8911
- Remove precision args in trainer due to PTL update by @yaoyu-33 :: PR: #8908
- Huvu/mcore retro by @huvunvidia :: PR: #8861
- fsdp tp > 1 bug fix by @dimapihtar :: PR: #8947
- Fix memory leak at loss func by @minitu :: PR: #8868
- change the condition for get qkv tensor from linear_qkv output in mcoremixin by @HuiyingLi :: PR: #8965
- Add safety checks for 'data' key in MegatronGPTModel cfg by @HuiyingLi :: PR: #8991
- [NeMo-UX] Adding MegatronParallel by @cuichenx :: PR: #8987
- Skip top_p computations when set to 1.0 by @odelalleau :: PR: #8905
- Gemma bug by @cuichenx :: PR: #8962
- [NeMo-UX] Adding megatron strategy by @marcromeyn :: PR: #8995
- Quantized checkpoint support in export and deploy modules by @janekl :: PR: #8859
- add geglu to mlp swap by @JRD971000 :: PR: #8999
- add timeout for new_group by @acphile :: PR: #8998
- Zero-shot evaluation pipeline for mcore RETRO by @huvunvidia :: PR: #8941
- Added fusion for squared relu by @sanandaraj5597 :: PR: #8963
- Developer Documents for mcore RETRO by @huvunvidia :: PR: #9026
- [NeMo-UX] Adding GPTModel & MockDataModule by @marcromeyn :: PR: #9011
- Adding unit test for mcore RETRO model by @huvunvidia :: PR: #9022
- docs and simplification of cmd args by @arendu :: PR: #8979
- [NeMo-UX] Add checkpoint-io to MegatronStrategy by @marcromeyn :: PR: #9057
- Enable Sequence Packing and Pipeline Parallel in NeVA by @yaoyu-33 :: PR: #8957
- Mingyuanm/add back fp8 support to sd by @Victor49152 :: PR: #9070
- unfused lora by @arendu :: PR: #9004
- Handle case where num_query_groups is set to null for LoRA config setup by @vysarge :: PR: #9075
- Alit/griffin by @JRD971000 :: PR: #9021
- Implement DistributedCheckpointIO by @mikolajblaz :: PR: #9016
- Video Neva Pretraining + Inference Implementation by @paul-gibbons :: PR: #9095
- HF to .nemo for Mixtral-8x22B-instruct by @akoumpa :: PR: #9060
- mcore ds updates by @dimapihtar :: PR: #8951
- Alit/griffin perf by @JRD971000 :: PR: #9107
- Add assert for max_steps to be positive in MegatronGPTSFTModel by @athitten :: PR: #9110
- Extend sequence length padding for GPT SFT to account for context parallel by @vysarge :: PR: #8869
- Update gpt dataset config parameter for mock by @thomasdhc :: PR: #9118
- Add Mcore DistributedDataParallel and distributed optimizer into Nemo by @gdengk :: PR: #9034
- Revert "Add assert for max_steps to be positive in MegatronGPTSFTMode… by @pablo-garay :: PR: #9128
- scripts to convert HF lora to nemo by @arendu :: PR: #9102
- Prevent duplicated checkpoints by @mikolajblaz :: PR: #9015
- add TN/ITN link in speech tools list by @erastorgueva-nv :: PR: #9142
- Cleanup deprecated files and temporary changes by @cuichenx :: PR: #9088
- Use DP+CP groups as the FSDP sharding domain by @erhoo82 :: PR: #9145
- CUDA memory profile by @erhoo82 :: PR: #9096
- Fix missing func for T5 model by @gdengk :: PR: #9141
- Add knob for load_directly_on_device by @mikolajblaz :: PR: #9125
- Revert rope fusion defaults by @cuichenx :: PR: #9238
- Update nemo.export module for quantized models by @janekl :: PR: #9250
- Fix circular import for MM dataprep notebook by @cuichenx :: PR: #9287
- neva media_type + text generation default fix by @paul-gibbons :: PR: #9257
- fix lora and ptuning and isort/black by @oyilmaz-nvidia :: PR: #9290
- add check if num layers is divisible by pp size by @dimapihtar :: PR: #9208
- Fix P-tuning for Llama based models by @apanteleev :: PR: #9297
- add deprecation warnings by @pablo-garay :: PR: #9266
- move pooler under post_process by @dimapihtar :: PR: #9328
- add deprecation note for nmt by @dimapihtar :: PR: #9342
- Fix incorrect checkpoint removal logic (#9192) by @mikolajblaz :: PR: #9204
- fix fp16 precision issue by @dimapihtar :: PR: #9376
- Fix module.training for Neva in FusedAttn backward which causes nan by @yaoyu-33 :: PR: #8877

</details>

#### Export

<details><summary>Changelog</summary>

- Updates for TRT-LLM 0.9 by @oyilmaz-nvidia :: PR: #8873
- Mingyuanm/sdxl export by @Victor49152 :: PR: #8926
- Avoid unpacking NeMo checkpoints before exporting to TRT-LLM by @apanteleev :: PR: #8866
- Update gemma for trt-llm 0.9 by @oyilmaz-nvidia :: PR: #8974
- TRT-LLM export P-tuning related fixes by @apanteleev :: PR: #8863

</details>

#### General Improvements

<details><summary>Changelog</summary>

- Update package info by @ericharper :: PR: #8793
- [Nemo CICD] Update mcore 4.13.24 by @pablo-garay :: PR: #8917
- Akoumparouli/low mem mixtral ckpt converter by @akoumpa :: PR: #8895
- Adding RETRO tests to Action Tests (cicd-main.yml)  by @huvunvidia :: PR: #8942
- Akoumparouli/fix sd train 2 by @akoumpa :: PR: #8883
- Update te install for jenkins by @ericharper :: PR: #8954
- [Nemo CICD] Add last job depending on others for blocking check by @pablo-garay :: PR: #8959
- Minor quantization pipeline updates by @janekl :: PR: #8924
- Fix External CLIP Converter by @yaoyu-33 :: PR: #8960
- PP support in LoRA merge script by @cuichenx :: PR: #8934
- Update PR template by @ericharper :: PR: #8978
- Update Latest News by @shashank3959 :: PR: #8837
- Fix incorrect link to latest news in README by @shashank3959 :: PR: #8985
- Update dependency install for LLM and MM by @ericharper :: PR: #8990
- Temporarily remove mcore dep by @ericharper :: PR: #9010
- [Nemo CICD] further specialize runners for more parallelism by @pablo-garay :: PR: #9036
- Update mm dataprep notebook based on feedback by @cuichenx :: PR: #9029
- Fix import in lora merge script by @cuichenx :: PR: #9032
- [Nemo CICD] Run when labeled:Run CICD by @pablo-garay :: PR: #9044
- [Nemo CICD] Add tag/label for 1-gpu runner by @pablo-garay :: PR: #9046
- [Nemo CICD] checkout v4 by @pablo-garay :: PR: #9048
- [Nemo CICD] Remove temp test change by @pablo-garay :: PR: #9049
- remove in-place addition for dreambooth train with text encoder by @Victor49152 :: PR: #8825
- Mingyuanm/sdxl quantization notebook by @Victor49152 :: PR: #9042
- [Nemo CICD] Trigger on comment issued by @pablo-garay :: PR: #9062
- zarr ckpt to torch_dist ckpt converter by @dimapihtar :: PR: #8842
- Restore PTQ tests for Llama2 (reopened) by @janekl :: PR: #9064
- add clip H config by @JRD971000 :: PR: #9082
- [NeMo-UX] Add mixed-precision plugin by @marcromeyn :: PR: #9065
- Comment baichuan test and update pr template by @ericharper :: PR: #9085
- Add safe extraction of nemo tar files by @athitten :: PR: #8976
- Improved `shard_id` parsing in `LazyNemoTarredIterator`, enables AIS dataloading by @pzelasko :: PR: #9077
- [NeMo-UX] Add mistral-7b model by @marcromeyn :: PR: #9066
- Llama3 Conversion Script Update by @suiyoubi :: PR: #9089
- dehardcode test string by @JimmyZhang12 :: PR: #8865
- [Nemo CICD] Try trigger cicd run on comment by @pablo-garay :: PR: #9111
- Lhotse dataloading: RIR augmentation and nemo/tarred input support for RIR and noise aug by @pzelasko :: PR: #9109
- mixtral evaluation PR by @Slyne :: PR: #8989
- [Nemo CICD] Revert: run GHA cicd on comment by @pablo-garay :: PR: #9119
- [Nemo CICD] Comment out flaky test: running too long by @pablo-garay :: PR: #9123
- [Nemo CICD] Add timeout to unit tests by @pablo-garay :: PR: #9132
- [Nemo CICD] Indicate optional test in name (prefix) by @pablo-garay :: PR: #9139
- video neva null image+video folder path fix by @paul-gibbons :: PR: #9116
- [NeMo-UX] Add data module by @cuichenx :: PR: #9133
- NeMo Inference Requirements by @oyilmaz-nvidia :: PR: #9093
- Remove debug print by @maanug-nv :: PR: #9074
- Remove legacy CI by @pablo-garay :: PR: #9149
- Update support for push_to_hf_hub() by @titu1994 :: PR: #9159
- [Nemo CICD] comment out flaky PTQ tests by @pablo-garay :: PR: #9160
- Update branch by @ericharper :: PR: #9211
- dist adam transpose fix by @dimapihtar :: PR: #9239
- [Nemo CICD] Increase time limit for Speech_Checkpoints_tests (#9186) by @pablo-garay :: PR: #9247
- Pin transformers by @ericharper :: PR: #9261
- Fix typo in HF tutorial by @titu1994 :: PR: #9302

</details>

## NVIDIA Neural Modules 1.23.0

### Highlights

#### Models

##### Nvidia Starcoder 2 - 15B

- Announcement - https://developer.nvidia.com/blog/unlock-your-llm-coding-potential-with-starcoder2/
- AI Foundation Model Inference  - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/starcoder2-15b
- https://huggingface.co/bigcode/starcoder2-15b

##### NeMo Canary
Announcement - https://nvidia.github.io/NeMo/blogs/2024/2024-02-canary/

- https://huggingface.co/nvidia/canary-1b

#### NeMo LLM

- Falcon
- Code Llama
- StarCoder
- GPT perf improvements
- Context parallelism
- Mistral
- Mixtral (without expert parallelism)
- Mcore GPT Dataset integration

#### NeMo MM
- CLIP
- Stable Diffusion (supporting LoRA)
- Imagen
- ControlNet (for SD)
- Instruct pix2pix (for SD)
- LLAVA
- NeVA
- DreamFusion++
- NSFW filtering

#### NeMo ASR

- Lhotse Dataloading support #7880
- Canary: Multi task multi lingual ASR #8242
- LongForm Audio for Diarization #7737
- Faster algorithm for RNN-T Greedy #7926
- Cache-Aware streaming notebook #8296

#### NeMo TTS

#### NeMo Vision

#### Known Issues

##### ASR

###### RNNT WER calculation when fused batch size > 1 during validation / test step()

Previously, the RNNT metric was stateful while the CTC one was not ([r1.22.0](https://github.com/NVIDIA/NeMo/blob/r1.22.0/nemo/collections/asr/metrics/rnnt_wer_bpe.py#L419-L420), [r1.23.0](https://github.com/NVIDIA/NeMo/blob/r1.23.0/nemo/collections/asr/metrics/wer.py#L333))

Therefore this calculation in the RNNT joint for fused operation worked properly. However with the unification of metrics in r1.23.0, a bug was introduced where only the last sub-batch of metrics calculates the scores and does not accumulate. This is patched via https://github.com/NVIDIA/NeMo/pull/8587 and will be fixed in the next release.

**Workaround**: Explicitly disable fused batch size during inference using the following command 

```python
from omegaconf import open_dict
model = ...
decoding_cfg = model.cfg.decoding
with open_dict(decoding_cfg):
  decoding_cfg.fused_batch_size = -1
model.change_decoding_strategy(decoding_cfg)
```

Note: This bug does not affect scores calculated via model.transcribe() (since it does not calculate metrics during inference, just text), or using the `transcribe_speech.py` or `speech_to_text_eval.py` in `examples/asr`.

###### Two failing unit tests due to a change in expected results, caused by lhotse version update

#### Container

For additional information regarding NeMo containers, please visit: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo

`docker pull nvcr.io/nvidia/nemo:24.01.speech`

#### ASR

<details><summary>Changelog</summary>

- Update link to yaml file in ASR_with_Transducers.ipynb by @Faith-Nchifor :: PR: #8014
- Use convert_hf_dataset_to_nemo by @karpnv :: PR: #8017
- Update asr_language_modeling.rst: Add a missing word by @martin0258 :: PR: #8007
- spelling mistake by @orena1 :: PR: #7903
- update asr eval by @stevehuang52 :: PR: #8045
- fix noise aug by @stevehuang52 :: PR: #8057
- Various fixes for typos and urls by @titu1994 :: PR: #8066
- [Fix] Increase length check tolerance to prevent test failing by @anteju :: PR: #8067
- Add text metrics to asr eval by @stevehuang52 :: PR: #8087
- fix device setting to allow using accelerator cpu by @orena1 :: PR: #8084
- .ctm in data simulator annotator compliant with RT-09 specification by @popcornell :: PR: #8004
- Fix AST eval by @stevehuang52 :: PR: #8112
- fix: numba.*_num_threads resets torch num_threads #8141 by @itzsimpl :: PR: #8145
- Update dependencies by @titu1994 :: PR: #8156
- NeMo + Lhotse integration by @pzelasko :: PR: #7880
- Speedup RNN-T greedy decoding by @artbataev :: PR: #7926
- [docker] Install k2 before NeMo for faster image rebuilding by @pzelasko :: PR: #8204
- [docs] Add --force_codec to tarred dataset creation examples by @pzelasko :: PR: #8227
- Temporarily use the previous RNN-T decoding algorithm as default by @artbataev :: PR: #8226
- Make TDT inference not require duration params by @hainan-xv :: PR: #8207
- Cache Aware Streaming tutorial notebook by @erastorgueva-nv :: PR: #8296
- fix path location and branch by @nithinraok :: PR: #8304
- Attention encoder-decoder models for multiple speech-to-text tasks  … by @titu1994 :: PR: #8324
- Remove asr webapp by @titu1994 :: PR: #8347
- remove _target_ at model level in aed model config [ASR] by @krishnacpuvvada :: PR: #8351
- Add change_vocabulary and save_tokenizers() support to Multitask ASR models by @titu1994 :: PR: #8357
- Change default beam size by @titu1994 :: PR: #8371
-  adding jenkins test for speech_to_text_aed model by @krishnacpuvvada :: PR: #8368
- Add Finetuning tutorial with HF Datasets by @nithinraok :: PR: #8356
- wer fix by @tbartley94 :: PR: #8404
- add ensemble decoding fix by @nithinraok :: PR: #8427
- Update k2 by @artbataev :: PR: #8492

</details>

#### TTS

<details><summary>Changelog</summary>

- [TTS] Scale sampler steps by number of devices by @rlangman :: PR: #7947
- Add All Multimodal Source Code Part 2: Text to image, x to nerf by @yaoyu-33 :: PR: #7970
- [TTS] Add period discriminator and feature matching loss to codec recipe by @rlangman :: PR: #7884
- Added VectorQuantizer base class by @anteju :: PR: #8011

</details>

#### LLMS

<details><summary>Changelog</summary>

- Add interface to set NCCL options of each process group by @erhoo82 :: PR: #7923
- Support O2 training of PEFT and SFT by @cuichenx :: PR: #7971
- [NLP] Access scaler only in FP16 case by @janekl :: PR: #7916
- [NLP] Minor improvements in Llama conversion script by @janekl :: PR: #7978
- [NLP] Use helpers from utils_funcs.py in Llama conversion by @janekl :: PR: #7979
- [NLP] Remove replace_sampler_ddp (deprecated in Trainer) by @janekl :: PR: #7981
- Reworked MegatronPretrainingRandomBatchSampler to correctly handle epochs > 1 by @trias702 :: PR: #7920
- Remove deprecated arguments from TE's TransformerLayer by @jbaczek :: PR: #7917
- Add All Multimodal Source Code by @yaoyu-33 :: PR: #7791
- First draft of mcore bert model in NeMo by @shanmugamr1992 :: PR: #7814
- Support Falcon Variants (7B/40B/180B) in Mcore NeMo by @xuanzic :: PR: #7666
- FSDP + Tensor Parallelism by @erhoo82 :: PR: #7897
- Packed Sequence by @cuichenx :: PR: #7945
- Adding method back that was removed accidentally by @ericharper :: PR: #8038
- [NLP] ArtifactItem with init=True to make it debuggable by @janekl :: PR: #7980
- SFT patch: (1) enable sequence parallelism and (2) enable profile by @erhoo82 :: PR: #7963
- migration to PTL 2.0 for spellmapper model by @bene-ges :: PR: #7924
- Change the megatron config lr scheduler default and fix to change partitions script by @shan18 :: PR: #8094
- (1) Add SHARP interface to M-CORE, (2) use send/recv to send train loss to the first rank instead of b-cast by @erhoo82 :: PR: #7793
- Reconfigure limit_val_batches only for int by @athitten :: PR: #8099
- Fixing wrapper and moving it to base class by @shanmugamr1992 :: PR: #8055
- fix gated_linear_unit bug by @Agoniii :: PR: #8042
- Fix Adapter for MCore models by @cuichenx :: PR: #8124
- add war fix for sync issues by @gshennvm :: PR: #8130
- Improve PEFT UX by @cuichenx :: PR: #8131
- Enhance flexibility by passing callbacks as method argument by @michal2409 :: PR: #8015
- context parallelism by @xrennvidia :: PR: #7739
- Make pipelined TP comm overlap available with mcore by @erhoo82 :: PR: #8005
- remove deprecated scripts by @arendu :: PR: #8138
- adding OnlineSampleMapping by @arendu :: PR: #8137
- Add distopt support for FP8 params and BF16 optimizer state by @timmoon10 :: PR: #7909
- Revert adding OnlineSampleMapping by @pablo-garay :: PR: #8164
- Token count and sequence length logging for MegatronGPTSFTModel by @vysarge :: PR: #8136
- Use latest apex internal API by @jbaczek :: PR: #8129
- tune specific params in the base model by @arendu :: PR: #7745
- Virtual pipeline parallel support for MegatronGPTSFTModel by @vysarge :: PR: #7964
- removed deprecated peft model by @arendu :: PR: #8183
- remove more deprecated files by @arendu :: PR: #8169
- Pre-generate cu_seqlens argmin and max_seqlen to remove host-to-device sync by @erhoo82 :: PR: #8108
- Add the interface to use SHARP to FSDP strategy by @erhoo82 :: PR: #8202
- Multimodal required NLP base model changes by @yaoyu-33 :: PR: #8188
- [NLP] Improve and unify loading state_dict for community models by @janekl :: PR: #7977
- Rename Finetuning Scripts by @cuichenx :: PR: #8201
- Final multimodal PR with our recent developments on MM side by @yaoyu-33 :: PR: #8127
- Add include_text parameter to SFT dataloaders by @Kipok :: PR: #8198
- Add random_seed argument to generate by @Kipok :: PR: #8162
- Added support for neptune logger by @harishankar-gopalan :: PR: #8210
- Pre-compute max_seqlen and cu_seqlens_argmin in all model-parallel cases by @erhoo82 :: PR: #8222
- Use PackedSeqParams in accordance with changes in Megatron-LM by @cuichenx :: PR: #8205
- Fix to peft & virtual pipeline parallel unsupported check by @vysarge :: PR: #8216
- Fixed the tp overlap switch by @sanandaraj5597 :: PR: #8195
- add knobs for rope/swiglu fusion by @lhb8125 :: PR: #8184
- Added sample cpu_offloading switch to YAML by @sanandaraj5597 :: PR: #8148
- Syncing random seed between ranks in generate by @Kipok :: PR: #8230
- add first_val_step to mcore scheduler by @JimmyZhang12 :: PR: #8150
- Correct padding for SFT input data to account for sequence parallel + TE's fp8 op dimension requirements by @vysarge :: PR: #8240
- Mistral 7b conversion script by @akoumpa :: PR: #8052
- switch to mcore dataset [with FIM support] by @dimapihtar :: PR: #8149
- Mixtral to NeMo conversion script. by @akoumpa :: PR: #8155
- fixes to accomendate mcore changes by @HuiyingLi :: PR: #8261
- Allow MegatronPretrainingRandomSampler to do multi-epoch training by @trias702 :: PR: #8239
- Add dist ckpt support for regular optimizers by @mikolajblaz :: PR: #7749
- add deallocate pipeline output optimization by @JimmyZhang12 :: PR: #8279
- Fix memory leak caused by context parallelism hanging references by omegaconf by @JimmyZhang12 :: PR: #8299
- distributed fused adam + rampup bs support by @dimapihtar :: PR: #8302
- Update PEFT Doc by @cuichenx :: PR: #8262
- Converter script fixes for mixtral/mistral by @akoumpa :: PR: #8272
- Keep max_seqlen and cu_seqlens_argmin for later micro-batches when PP>1 by @erhoo82 :: PR: #8334
- Enable megatron core loggers for GPT pretraining by @ashbhandare :: PR: #8354
- mcore ds fix by @dimapihtar :: PR: #8283
- release updates by @dimapihtar :: PR: #8378
- Mcore customization doc by @HuiyingLi :: PR: #8298
- updated link to pubmed by @nithinraok :: PR: #8402
- mcore customization doc minor fix by @HuiyingLi :: PR: #8421
- Fixing mcore bert for TP, PP and SP by @shanmugamr1992 :: PR: #8336
- Add settings to suppress bf16 compile errors in CI on V100 by @athitten :: PR: #8481
- MoE parameter passing by @akoumpa :: PR: #8255
- Add fp8 support for SD/Update notebook paths by @Victor49152 :: PR: #8489

</details>

#### NeMo Tools

<details><summary>Changelog</summary>

- SDE bugfix log by @Jorjeous :: PR: #8430

</details>

#### General Improvements

<details><summary>Changelog</summary>

- Add news section to README by @ericharper :: PR: #7984
- Fixing conversion script to work for code llama by @shanmugamr1992 :: PR: #7997
- Fix crash when converting to mcore a model using rotary embeddings by @odelalleau :: PR: #7998
- Added a procedure for Windows users, README by @Jorjeous :: PR: #7942
- Update manifest.py to speedup loading tarred datasets by @stevehuang52 :: PR: #7900
- [Fix] Fixed name of a test by @anteju :: PR: #7986
- Fix lora merge script by @cuichenx :: PR: #8113
- Support transcoding audio formats when saving tarred datasets (FLAC, OPUS) by @pzelasko :: PR: #8102
- README edit to change Apple Silicon install instructions (to fix a break introduced by pytorch 2) by @stephenmcconnachie :: PR: #8122
- Fixes NVIDIA/apex installation to not erroneously install the  pkg by @terrykong :: PR: #8126
- Graphviz fix by @GNroy :: PR: #7843
- Update README.rst by @fayejf :: PR: #8154
- Fix TP>1 issue for conversion script by @cuichenx :: PR: #8144
- Support torch jit script by @artbataev :: PR: #8027
- NeMo Multimodal Docs and Tests Initial PR by @yaoyu-33 :: PR: #8028
- Remove left-over prints in NeMo+Lhotse code by @pzelasko :: PR: #8180
- Upgrade to DLFW PyTorch 23.12 by @ericharper :: PR: #8163
- Add Lhotse support for  key in NeMo manifests by @pzelasko :: PR: #8197
- Fix CPU Initialization and TP>1 for LoRA Merge Script by @cuichenx :: PR: #8199
- Add support in Neural Typecheck to disable semantic checks by @titu1994 :: PR: #8212
- Pin lhotse=1.19.2 in r1.23.0 by @pzelasko :: PR: #8303
- Multimodal r1.23.0 bug fix  by @yaoyu-33 :: PR: #8315
- MCore dataset compatibility for tokenizers by @vysarge :: PR: #8390
- Update NFA video download link by @erastorgueva-nv :: PR: #8406
- Update MM Dataprep Tutorial by @cuichenx :: PR: #8410
- Fix dreambooth data sampler issue by @yaoyu-33 :: PR: #8400
- Fix a bug in CTM line processing function for multi-speaker data simulations by @tango4j :: PR: #8416
- Akoumparouli/mistral bugfix by @akoumpa :: PR: #8353
- pin to 0.5.0 by @ericharper :: PR: #8465
- Update NeMo Multimodal Requirements by @yaoyu-33 :: PR: #8515
- Fix link in multimodal dataprep tutorial by @cuichenx :: PR: #8517

</details>
