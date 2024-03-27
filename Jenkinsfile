pipeline {
  agent {
        docker {
          image 'nvcr.io/nvidia/pytorch:24.01-py3'
          args '--device=/dev/nvidia0 --gpus all --user 0:128 -v /home/TestData:/home/TestData -v $HOME/.cache:/root/.cache --shm-size=8g --env TRANSFORMERS_OFFLINE=0 --env HYDRA_FULL_ERROR=1'
        }
  }

  environment {
        NVTE_FUSED_ATTN = 0
        NVTE_FLASH_ATTN = 0
  }

  options {
    timeout(time: 8, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: true)
  }

  stages {

    stage('Add git safe directory'){
      steps{
        sh 'git config --global --add safe.directory /var/lib/jenkins/workspace/NeMo_$GIT_BRANCH'
        sh 'git config --global --add safe.directory /raid/JenkinsWorkDir/workspace/NeMo_$GIT_BRANCH'
        sh 'git config --global --add safe.directory /mnt/D3/JenkinsWorkDir/workspace/NeMo_$GIT_BRANCH'
      }
    }

    stage('nvidia-smi'){
      steps{
        sh 'nvidia-smi'
      }
    }

    stage('PyTorch version') {
      steps {
        sh 'python -c "import torch; print(torch.__version__)"'
        sh 'python -c "import torchvision; print(torchvision.__version__)"'
      }
    }

    stage('Install test requirements') {
      steps {
        sh 'apt-get update && apt-get install -y bc && pip install -r requirements/requirements_test.txt && pip install -r requirements/requirements_lightning.txt'
      }
    }

    stage('Code formatting checks') {
      steps {
        sh 'python setup.py style'
      }
    }

    stage('Copyright Headers check') {
      steps {
        sh 'python tests/check_copyright_header.py --dir .'
      }
    }

    stage('NeMo Installation') {
      steps {
        sh './reinstall.sh release'
      }
    }

    // Transformer Engine 1.2.0
    stage('Transformer Engine installation') {
      steps {
         sh 'git clone https://github.com/NVIDIA/TransformerEngine.git && \
             cd TransformerEngine && \
             git fetch origin 8c9abbb80dba196f086b8b602a7cf1bce0040a6a && \
             git checkout FETCH_HEAD && \
             git submodule init && git submodule update && \
             NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .'
      }
    }

    // Apex bugfix for PyTorch 23.11 container: https://github.com/NVIDIA/apex/pull/1760
    stage('Apex installation') {
      steps {
         sh 'git clone https://github.com/NVIDIA/apex.git && \
             cd apex && \
             git checkout c07a4cf67102b9cd3f97d1ba36690f985bae4227 && \
             cp -R apex /usr/local/lib/python3.10/dist-packages'
      }
    }

    stage('Pytorch lightning installation') {
      steps {
         sh 'git clone -b bug_fix https://github.com/athitten/pytorch-lightning.git && \
             cd pytorch-lightning && \
             PACKAGE_NAME=pytorch pip install -e .'
      }
    }

    // pip package should be working with main, if not we can update the commit here
    // until the pip package is updated
    stage('Megatron Core installation') {
      steps {
         sh 'git clone https://github.com/NVIDIA/Megatron-LM.git && \
             cd Megatron-LM && \
             git checkout a5415fcfacef2a37416259bd38b7c4b673583675 && \
             pip install .'
      }
    }

    stage('AMMO installation') {
      steps {
         sh 'pip install nvidia-ammo~=0.7.0 --extra-index-url https://pypi.nvidia.com --no-cache-dir'
      }
    }

    stage('PyTorch Lightning version') {
      steps {
        sh 'python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"'
      }
    }

    stage('PyTorch Lightning DDP Checks') {
      steps {
        sh 'CUDA_VISIBLE_DEVICES="0,1" python "tests/core_ptl/check_for_ranks.py"'
      }
    }

    stage('Basic Import Checks') {
      steps {
        sh 'python -c "import nemo.collections.asr as nemo_asr"'
        sh 'python -c "import nemo.collections.nlp as nemo_nlp"'
        sh 'python -c "import nemo.collections.tts as nemo_tts"'
      }
    }
    stage('Import Checks'){
      steps {
        sh 'python tests/core_ptl/check_imports.py --domain "nlp"'
      }
    }
    stage('L0: Unit Tests GPU') {
      steps {
        sh 'NEMO_NUMBA_MINVER=0.53 pytest -m "not pleasefixme" --with_downloads'
      }
    }

   stage('L0: Unit Tests CPU') {
     when {
       anyOf {
         branch 'main'
         changeRequest target: 'main'
       }
     }
     steps {
       sh 'CUDA_VISIBLE_DEVICES="" NEMO_NUMBA_MINVER=0.53 pytest -m "not pleasefixme" --cpu --with_downloads --relax_numba_compat'
     }
   }

    stage('L2: Multimodal Imagen Train') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/multimodal/imagen_train"
        sh "python examples/multimodal/text_to_image/imagen/imagen_training.py \
        trainer.precision=16 \
        trainer.num_nodes=1 \
        trainer.devices=1 \
        ++exp_manager.max_time_per_run=00:00:03:00 \
        trainer.max_steps=20 \
        model.conditioning.embed_dim=64 \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.synthetic_data=True \
        exp_manager.exp_dir=/home/TestData/multimodal/imagen_train \
        model.inductor=False \
        model.unet.flash_attention=False \
        "
        sh "rm -rf /home/TestData/multimodal/imagen_train"
      }
    }

    stage('L2: Multimodal Stable Diffusion Train') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/multimodal/stable_diffusion_train"
        sh "python examples/multimodal/text_to_image/stable_diffusion/sd_train.py \
            trainer.precision=16 \
            trainer.num_nodes=1 \
            trainer.devices=1 \
            ++exp_manager.max_time_per_run=00:00:03:00 \
            trainer.max_steps=20 \
            model.micro_batch_size=1 \
            model.global_batch_size=1 \
            model.data.synthetic_data=True \
            exp_manager.exp_dir=/home/TestData/multimodal/stable_diffusion_train \
            model.inductor=False \
            model.cond_stage_config._target_=nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenCLIPEmbedder \
            ++model.cond_stage_config.version=openai/clip-vit-large-patch14 \
            ++model.cond_stage_config.max_length=77 \
            ~model.cond_stage_config.restore_from_path \
            ~model.cond_stage_config.freeze \
            ~model.cond_stage_config.layer \
            model.unet_config.from_pretrained=null \
            model.first_stage_config.from_pretrained=null \
            model.unet_config.use_flash_attention=False \
            model.unet_config.attention_resolutions=[1] \
            model.unet_config.channel_mult=[1] \
            "
        sh "rm -rf /home/TestData/multimodal/stable_diffusion_train"
      }
    }
//     stage('L2: Multimodal ControlNet Train') {
//       when {
//         anyOf {
//           branch 'main'
//           changeRequest target: 'main'
//         }
//       }
//       failFast true
//       steps {
//         sh "rm -rf /home/TestData/multimodal/controlnet_train"
//         sh "python examples/multimodal/text_to_image/controlnet/controlnet_train.py \
//             trainer.precision=16 \
//             trainer.num_nodes=1 \
//             trainer.devices=1 \
//             ++exp_manager.max_time_per_run=00:00:03:00 \
//             trainer.max_steps=20 \
//             model.micro_batch_size=1 \
//             model.global_batch_size=1 \
//             model.data.synthetic_data=True \
//             exp_manager.exp_dir=/home/TestData/multimodal/controlnet_train \
//             model.inductor=False \
//             model.image_logger.max_images=0 \
//             model.control_stage_config.params.from_pretrained_unet=null \
//             model.unet_config.from_pretrained=null \
//             model.first_stage_config.from_pretrained=null \
//             model.unet_config.use_flash_attention=False \
//             "
//         sh "rm -rf /home/TestData/multimodal/controlnet_train"
//       }
//     }
//     stage('L2: Multimodal DreamBooth Train') {
//       when {
//         anyOf {
//           branch 'main'
//           changeRequest target: 'main'
//         }
//       }
//       failFast true
//       steps {
//         sh "rm -rf /home/TestData/multimodal/dreambooth_train"
//         sh "python examples/multimodal/text_to_image/dreambooth/dreambooth.py \
//             trainer.precision=16 \
//             trainer.num_nodes=1 \
//             trainer.devices=1 \
//             ++exp_manager.max_time_per_run=00:00:03:00 \
//             trainer.max_steps=20 \
//             model.micro_batch_size=1 \
//             model.global_batch_size=1 \
//             exp_manager.exp_dir=/home/TestData/multimodal/dreambooth_train \
//             model.inductor=False \
//             model.cond_stage_config._target_=nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenCLIPEmbedder \
//             ++model.cond_stage_config.version=openai/clip-vit-large-patch14 \
//             ++model.cond_stage_config.max_length=77 \
//             ~model.cond_stage_config.restore_from_path \
//             ~model.cond_stage_config.freeze \
//             ~model.cond_stage_config.layer \
//             model.unet_config.from_pretrained=null \
//             model.first_stage_config.from_pretrained=null \
//             model.data.instance_dir=/home/TestData/multimodal/tiny-dreambooth \
//             model.unet_config.use_flash_attention=False \
//             "
//         sh "rm -rf /home/TestData/multimodal/dreambooth_train"
//       }
//     }
    stage('L2: Vision ViT Pretrain TP=1') {
      when {
        anyOf {
          branch 'r1.23.0'
          changeRequest target: 'r1.23.0'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/vision/vit_pretrain_tp1"
        sh "python examples/vision/vision_transformer/megatron_vit_classification_pretrain.py \
            trainer.precision=16 \
            model.megatron_amp_O2=False \
            trainer.num_nodes=1 \
            trainer.devices=1 \
            trainer.val_check_interval=5 \
            ++exp_manager.max_time_per_run=00:00:03:00 \
            trainer.max_steps=20 \
            model.micro_batch_size=2 \
            model.global_batch_size=4 \
            model.tensor_model_parallel_size=1 \
            model.pipeline_model_parallel_size=1 \
            model.data.num_workers=0 \
            exp_manager.create_checkpoint_callback=False \
            model.data.data_path=[/home/TestData/multimodal/tiny-imagenet/train,/home/TestData/multimodal/tiny-imagenet/val] \
            exp_manager.exp_dir=/home/TestData/vision/vit_pretrain_tp1 "
        sh "rm -rf /home/TestData/vision/vit_pretrain_tp1"
      }
    }

    stage('L2: Multimodal CLIP Pretrain TP=1') {
      when {
        anyOf {
          branch 'r1.23.0'
          changeRequest target: 'r1.23.0'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/multimodal/clip_pretrain_tp1"
        sh "python examples/multimodal/vision_language_foundation/clip/megatron_clip_pretrain.py  \
            trainer.precision=16 \
            model.megatron_amp_O2=False \
            trainer.num_nodes=1 \
            trainer.devices=1 \
            trainer.val_check_interval=10 \
            ++exp_manager.max_time_per_run=00:00:03:00 \
            trainer.max_steps=20 \
            model.micro_batch_size=1 \
            model.global_batch_size=1 \
            model.tensor_model_parallel_size=1 \
            model.pipeline_model_parallel_size=1 \
            exp_manager.create_checkpoint_callback=False \
            model.data.num_workers=0 \
            model.vision.num_layers=2 \
            model.text.num_layers=2 \
            model.vision.patch_dim=32 \
            model.vision.encoder_seq_length=49 \
            model.vision.class_token_length=7 \
            model.data.train.dataset_path=[/home/TestData/multimodal/tiny-clip/00000.tar] \
            model.data.validation.dataset_path=[/home/TestData/multimodal/tiny-clip/00000.tar] \
            model.data.webdataset.local_root_path=/ \
            exp_manager.exp_dir=/home/TestData/multimodal/clip_pretrain_tp1 "
        sh "rm -rf /home/TestData/multimodal/clip_pretrain_tp1"
      }
    }

    stage('L2: Multimodal NeVA Pretrain TP=1') {
      when {
        anyOf {
          branch 'r1.23.0'
          changeRequest target: 'r1.23.0'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/multimodal/neva_pretrain_tp1"
        sh "python examples/multimodal/multimodal_llm/neva/neva_pretrain.py \
            trainer.precision=16 \
            model.megatron_amp_O2=False \
            trainer.num_nodes=1 \
            trainer.devices=1 \
            trainer.val_check_interval=10 \
            trainer.limit_val_batches=5 \
            trainer.log_every_n_steps=1 \
            ++exp_manager.max_time_per_run=00:00:03:00 \
            trainer.max_steps=20 \
            model.micro_batch_size=2 \
            model.global_batch_size=4 \
            model.tensor_model_parallel_size=1 \
            model.pipeline_model_parallel_size=1 \
            exp_manager.create_checkpoint_callback=False \
            model.data.data_path=/home/TestData/multimodal/tiny-neva/dummy.json \
            model.data.image_folder=/home/TestData/multimodal/tiny-neva/images \
            model.tokenizer.library=sentencepiece \
            model.tokenizer.model=/home/TestData/multimodal/tiny-neva/tokenizer_add_special.model \
            model.num_layers=2 \
            model.hidden_size=5120 \
            model.ffn_hidden_size=13824 \
            model.num_attention_heads=40 \
            model.normalization=rmsnorm \
            model.data.num_workers=0 \
            model.data.conv_template=llama_2 \
            model.mm_cfg.vision_encoder.from_pretrained='openai/clip-vit-large-patch14' \
            model.mm_cfg.llm.from_pretrained=null \
            model.use_flash_attention=false \
            exp_manager.exp_dir=/home/TestData/multimodal/neva_pretrain_tp1 "
        sh "rm -rf /home/TestData/multimodal/neva_pretrain_tp1"
      }
    }

    stage('Setup test data and models') {
      steps {
        sh 'python -m tests.setup --save_dir /home/TestData/nlp'
      }
    }

    // TODO: this requires TE >= v0.11 which is not available in 23.06.
    //        please uncomment this test once mcore CI is ready.
    

    stage('L2: Community LLM Checkpoints tests') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Llama') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES=0 python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
            --input_name_or_path=/home/TestData/nlp/megatron_llama/llama-ci-hf \
            --output_path=/home/TestData/nlp/megatron_llama/llama_ci.nemo'
          }
        }
        stage('StarCoder') {
          steps {
            sh 'python scripts/checkpoint_converters/convert_starcoder_hf_to_nemo.py \
            --input_name_or_path /home/TestData/nlp/megatron_gpt/starcoder-ci-hf \
            --output_path /home/TestData/nlp/megatron_gpt/starcoder-ci-hf'
            sh 'rm -f /home/TestData/nlp/megatron_gpt/starcoder-ci-hf/megatron_starcoder_tp1_pp1.nemo'
          }
        }
        stage('Falcon') {
          steps {
            sh 'python scripts/checkpoint_converters/convert_falcon_hf_to_nemo.py \
            --input_name_or_path /home/TestData/nlp/megatron_gpt/falcon-ci-hf \
            --output_path /home/TestData/nlp/megatron_gpt/falcon-ci-hf/falcon_ci.nemo'
            sh 'rm -f /home/TestData/nlp/megatron_gpt/falcon-ci-hf/falcon_ci.nemo'
          }
        }
        stage('Baichuan2') {
          steps {
            sh 'python scripts/checkpoint_converters/convert_baichuan2_hf_to_nemo.py \
            --input_name_or_path=/home/TestData/nlp/megatron_gpt/Baichuan2-7B-Base \
            --output_path=/home/TestData/nlp/megatron_gpt/Baichuan2-7B-Base/ci.nemo'
            sh 'rm -f /home/TestData/nlp/megatron_gpt/Baichuan2-7B-Base/ci.nemo'
          }
        }
      }
    }

    stage('L2: Nemo PTQ') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Llama2 - Export Only') {
          steps {
            sh 'python examples/nlp/language_modeling/megatron_llama_quantization.py \
            model_file=/home/TestData/nlp/megatron_llama/llama_ci.nemo \
            quantization.algorithm=null \
            model_save=/home/TestData/nlp/megatron_llama/ci_baseline'
            sh 'rm -rf /home/TestData/nlp/megatron_llama/ci_baseline'
          }
        }
        stage('Llama2 - INT8 SQ') {
          steps {
            sh 'python examples/nlp/language_modeling/megatron_llama_quantization.py \
            model_file=/home/TestData/nlp/megatron_llama/llama_ci_megatron_amp_O2_hf_tokenizer.nemo \
            quantization.calib_dataset=/home/TestData/nlp/test_quantization/test.json \
            quantization.algorithm=int8_sq \
            quantization.num_calib_size=8 \
            inference.batch_size=2 \
            model_save=/home/TestData/nlp/megatron_llama/ci_int8_sq.qnemo'
            sh 'rm -f /home/TestData/nlp/megatron_llama/ci_int8_sq.qnemo'
          }
        }
        stage('Llama2 - FP8') {
          steps {
            sh 'python examples/nlp/language_modeling/megatron_llama_quantization.py \
            model_file=/home/TestData/nlp/megatron_llama/llama_ci.nemo \
            tensor_model_parallel_size=2 \
            trainer.devices=2 \
            quantization.calib_dataset=/home/TestData/nlp/test_quantization/test.json \
            quantization.algorithm=fp8 \
            quantization.num_calib_size=8 \
            inference.batch_size=2 \
            export.inference_tensor_parallel=2 \
            model_save=/home/TestData/nlp/megatron_llama/ci_fp8.qnemo'
            sh 'rm -f /home/TestData/nlp/megatron_llama/ci_fp8.qnemo'
          }
        }
      }
    }

    stage('L2: ASR dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_results'
            sh 'rm -rf examples/asr/speech_to_text_results'
          }
        }

        stage('Speech to Text WPE - CitriNet') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
            --config-path="../conf/citrinet/" --config-name="config_bpe" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
            model.tokenizer.type="wpe" \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_results'
            sh 'rm -rf examples/asr/speech_to_text_wpe_results'
          }
        }

        stage('Speech Pre-training - CitriNet') {
          steps {
            sh 'python examples/asr/speech_pretraining/speech_pre_training.py \
            --config-path="../conf/ssl/citrinet/" --config-name="citrinet_ssl_ci" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_pre_training_results'
            sh 'rm -rf examples/asr/speech_pre_training_results'
          }
        }

        stage('Speech To Text Finetuning') {
          steps {
            sh 'python examples/asr/speech_to_text_finetune.py \
            --config-path="conf/asr_finetune" --config-name="speech_to_text_finetune" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            init_from_nemo_model=/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo \
            model.tokenizer.update_tokenizer=False \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_finetuning_results'
            sh 'rm -rf examples/asr/speech_finetuning_results'
          }
        }

        stage('Speech To Text HF Finetuning') {
          steps {
            sh 'python examples/asr/speech_to_text_finetune.py \
            --config-path="conf/asr_finetune" --config-name="speech_to_text_hf_finetune" \
            ~model.train_ds.hf_data_cfg \
            model.train_ds.num_workers=1 \
            model.train_ds.batch_size=2 model.validation_ds.batch_size=2 \
            model.train_ds.streaming=true \
            +model.train_ds.hf_data_cfg.path="librispeech_asr" \
            +model.train_ds.hf_data_cfg.name=null \
            +model.train_ds.hf_data_cfg.split="test.clean" \
            +model.train_ds.hf_data_cfg.streaming=true \
            ~model.validation_ds.hf_data_cfg \
            model.validation_ds.streaming=true \
            +model.validation_ds.hf_data_cfg.path="librispeech_asr" \
            +model.validation_ds.hf_data_cfg.name=null \
            +model.validation_ds.hf_data_cfg.split="test.clean" \
            +model.validation_ds.hf_data_cfg.streaming=true \
            ~model.test_ds \
            init_from_nemo_model=/home/TestData/asr/stt_en_fastconformer_transducer_large.nemo \
            model.tokenizer.update_tokenizer=False \
            model.optim.sched.warmup_steps=0 \
            +model.optim.sched.max_steps=3 \
            trainer.max_epochs=null \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_finetuning_results'
            sh 'rm -rf examples/asr/speech_finetuning_results'
          }
        }

        // TODO: Please Fix Me
        // Error locating target 'nemo.collections.asr.modules.wav2vec_modules.ConvFeatureEncoder', see chained exception above.
        // stage('L2: Speech Pre-training - Wav2Vec') {
        //   steps {
        //     sh 'python examples/asr/speech_pretraining/speech_pre_training.py \
        //     --config-path="../conf/ssl/wav2vec/" --config-name="wav2vec_ci" \
        //     model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
        //     model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
        //     trainer.devices=[1] \
        //     trainer.accelerator="gpu" \
        //     +trainer.fast_dev_run=True \
        //     exp_manager.exp_dir=examples/asr/speech_pre_training_results'
        //     sh 'rm -rf examples/asr/speech_pre_training_results'
        //   }
        // }

        stage('L2: Speech to Text WPE - Conformer') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
            --config-path="../conf/conformer" --config-name="conformer_ctc_bpe" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
            model.tokenizer.type="wpe" \
            model.train_ds.batch_size=4 \
            model.validation_ds.batch_size=4 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_conformer_results'
            sh 'rm -rf examples/asr/speech_to_text_wpe_conformer_results'
          }
        }
      }
    }

    stage('L2: ASR dev run - part two') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Speech to Text WPE - Squeezeformer') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
            --config-path="../conf/squeezeformer" --config-name="squeezeformer_ctc_bpe" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
            model.tokenizer.type="wpe" \
            model.encoder.d_model=144 \
            model.train_ds.batch_size=4 \
            model.validation_ds.batch_size=4 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_squeezeformer_results'
            sh 'rm -rf examples/asr/speech_to_text_wpe_squeezeformer_results'
          }
        }
      }
    }

    stage('L2: Speech to Text EMA') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      steps {
        sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
        model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
        model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
        trainer.devices=2 \
        trainer.accelerator="gpu" \
        +trainer.fast_dev_run=True \
        +exp_manager.ema.enable=True \
        exp_manager.exp_dir=examples/asr/speech_to_text_results'
        sh 'rm -rf examples/asr/speech_to_text_results'
      }

    }

    stage('L2: Speech to Text AED') {
      when {
        anyOf {
          branch 'r1.23.0'
          changeRequest target: 'r1.23.0'
        }
      }
      steps {
        sh 'python examples/asr/speech_multitask/speech_to_text_aed.py \
        model.prompt_format=canary \
        model.model_defaults.asr_enc_hidden=256 \
        model.model_defaults.lm_dec_hidden=256 \
        model.encoder.n_layers=12 \
        model.transf_encoder.num_layers=0 \
        model.transf_decoder.config_dict.num_layers=12 \
        model.train_ds.manifest_filepath=/home/TestData/asr/manifests/canary/an4_canary_train.json \
        ++model.train_ds.is_tarred=false \
        model.train_ds.batch_duration=60 \
        +model.train_ds.text_field="answer" \
        +model.train_ds.lang_field="target_lang" \
        model.validation_ds.manifest_filepath=/home/TestData/asr/manifests/canary/an4_canary_val.json \
        +model.validation_ds.text_field="answer" \
        +model.validation_ds.lang_field="target_lang" \
        model.test_ds.manifest_filepath=/home/TestData/asr/manifests/canary/an4_canary_val.json \
        +model.test_ds.text_field="answer" \
        +model.test_ds.lang_field="target_lang" \
        model.tokenizer.langs.spl_tokens.dir=/home/TestData/asr_tokenizers/canary/canary_spl_tokenizer_v32 \
        model.tokenizer.langs.spl_tokens.type="bpe" \
        model.tokenizer.langs.en.dir=/home/TestData/asr_tokenizers/canary/en/tokenizer_spe_bpe_v1024_max_4 \
        model.tokenizer.langs.en.type=bpe \
        ++model.tokenizer.langs.es.dir=/home/TestData/asr_tokenizers/canary/es/tokenizer_spe_bpe_v1024_max_4 \
        ++model.tokenizer.langs.es.type=bpe \
        trainer.devices=[0] \
        trainer.accelerator="gpu" \
        +trainer.use_distributed_sampler=false \
        +trainer.fast_dev_run=True \
        exp_manager.exp_dir=examples/asr/speech_to_text_aed_results'
        sh 'rm -rf examples/asr/speech_to_text_results'
      }

    }

    stage('L2: Speaker dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speaker Recognition') {
          steps {
            sh 'python examples/speaker_tasks/recognition/speaker_reco.py \
            model.train_ds.batch_size=10 \
            model.validation_ds.batch_size=2 \
            model.train_ds.manifest_filepath=/home/TestData/an4_speaker/train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_speaker/dev.json \
            model.decoder.num_classes=2 \
            trainer.max_epochs=10 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/speaker_tasks/recognition/speaker_recognition_results'
            sh 'rm -rf examples/speaker_tasks/recognition/speaker_recognition_results'
          }
        }

        stage('Speaker Diarization') {
          steps {
            sh 'python examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder.py \
            model.diarizer.speaker_embeddings.model_path=titanet_large \
            model.train_ds.batch_size=5 \
            model.validation_ds.batch_size=5 \
            model.train_ds.emb_dir=examples/speaker_tasks/diarization/speaker_diarization_results \
            model.validation_ds.emb_dir=examples/speaker_tasks/diarization/speaker_diarization_results \
            model.train_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_train/msdd_data.50step.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_diarizer/simulated_valid/msdd_data.50step.json \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/speaker_tasks/diarization/speaker_diarization_results'
            sh 'rm -rf examples/speaker_tasks/diarization/speaker_diarization_results'
          }
        }

        stage('Speech to Label') {
          steps {
            sh 'python examples/asr/speech_classification/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            model.test_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            model.preprocessor._target_=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            ~model.preprocessor.window_size \
            ~model.preprocessor.window_stride \
            ~model.preprocessor.window \
            ~model.preprocessor.n_mels \
            ~model.preprocessor.n_mfcc \
            ~model.preprocessor.n_fft \
            exp_manager.exp_dir=examples/asr/speech_to_label_results'
            sh 'rm -rf examples/asr/speech_to_label_results'
          }
        }

        stage('Speaker Diarization with ASR Inference') {
          steps {
            sh 'python examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_with_asr_infer.py \
	        diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
            diarizer.speaker_embeddings.model_path=/home/TestData/an4_diarizer/spkr.nemo \
            diarizer.speaker_embeddings.parameters.save_embeddings=True \
            diarizer.speaker_embeddings.parameters.window_length_in_sec=[1.5] \
            diarizer.speaker_embeddings.parameters.shift_length_in_sec=[0.75] \
            diarizer.speaker_embeddings.parameters.multiscale_weights=[1.0] \
            diarizer.asr.model_path=QuartzNet15x5Base-En \
            diarizer.asr.parameters.asr_based_vad=True \
            diarizer.out_dir=examples/speaker_tasks/diarization/speaker_diarization_asr_results'
            sh 'rm -rf examples/speaker_tasks/diarization/speaker_diarization_asr_results'
          }
        }

        stage('Clustering Diarizer Inference') {
          steps {
            sh 'python examples/speaker_tasks/diarization/clustering_diarizer/offline_diar_infer.py \
	        diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
            diarizer.speaker_embeddings.model_path=/home/TestData/an4_diarizer/spkr.nemo \
            diarizer.speaker_embeddings.parameters.save_embeddings=True \
            diarizer.speaker_embeddings.parameters.window_length_in_sec=1.5 \
            diarizer.speaker_embeddings.parameters.shift_length_in_sec=0.75 \
            diarizer.speaker_embeddings.parameters.multiscale_weights=null \
            diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
            diarizer.out_dir=examples/speaker_tasks/diarization/clustering_diarizer_results'
            sh 'rm -rf examples/speaker_tasks/diarization/clustering_diarizer_results'
          }
        }

        stage('Neural Diarizer Inference') {
          steps {
            sh 'python examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder_infer.py \
            diarizer.manifest_filepath=/home/TestData/an4_diarizer/an4_manifest.json \
            diarizer.msdd_model.model_path=/home/TestData/an4_diarizer/diar_msdd_telephonic.nemo \
            diarizer.speaker_embeddings.parameters.save_embeddings=True \
            diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
            diarizer.out_dir=examples/speaker_tasks/diarization/neural_diarizer_results'
            sh 'rm -rf examples/speaker_tasks/diarization/neural_diarizer_results'
          }
        }

        stage('Multispeaker ASR Data Simulation') {
          steps {
            sh 'python tools/speech_data_simulator/multispeaker_simulator.py \
            --config-path=conf --config-name=data_simulator.yaml \
            data_simulator.random_seed=42 \
            data_simulator.manifest_filepath=/home/TestData/LibriSpeechShort/dev-clean-align-short.json \
            data_simulator.outputs.output_dir=./test_simulator \
            data_simulator.session_config.num_sessions=2 \
            data_simulator.session_config.session_length=60'
            sh 'rm -rf ./test_simulator'
          }
        }
      }
    }
    // TODO: Enable test after 21.08 container is used.
    // stage('L2: ASR DALI dev run') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('Speech to Text - DALI AudioToMelSpectrogramPreprocessor') {
    //       steps {
    //         sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         +model.train_ds.use_dali=True \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         +model.validation_ds.use_dali=True \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_results'
    //         sh 'rm -rf examples/asr/speech_to_text_results'
    //       }
    //     }
    //    stage('Speech to Text BPE - DALI AudioToMelSpectrogramPreprocessor') {
    //       steps {
    //         sh 'python examples/asr/asr_ctc/speech_to_text_bpe.py \
    //         --config-path="../conf/citrinet/" --config-name="config_bpe" \
    //         model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
    //         model.tokenizer.type="wpe" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         +model.train_ds.use_dali=True \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         +model.validation_ds.use_dali=True \
    // 	       trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_wpe_results'
    //         sh 'rm -rf examples/asr/speech_to_text_wpe_results'
    //       }
    //     }
    //     // TODO: This would fail due to an unnecessary torchaudio import.
    //     //       To be enabled once torchaudio is available in the container used for CI
    //     // stage('Speech to Text - DALI AudioToMFCCPreprocessor') {
    //     //   steps {
    //     //     sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
    //     //     model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //     //     +model.train_ds.use_dali=True \
    //     //     model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //     //     +model.validation_ds.use_dali=True \
    //     //     model.preprocessor._target_=nemo.collections.asr.modules.AudioToMFCCPreprocessor \
    //     //     ~model.preprocessor.normalize \
    //     //     ~model.preprocessor.features \
    //     //     ~model.preprocessor.frame_splicing \
    //     //     ~model.preprocessor.dither \
    //     //     ~model.preprocessor.stft_conv \
    //     //     +model.n_mels=64 \
    //     //     +model.n_mfcc=64 \
    //     //     trainer.devices=[1] \
    //     //     trainer.accelerator="gpu" \
    //     //     +trainer.fast_dev_run=True \
    //     //     exp_manager.exp_dir=examples/asr/speech_to_text_results'
    //     //     sh 'rm -rf examples/asr/speech_to_text_results'
    //     //   }
    //     // }
    //   }
    // }

    // TODO: Add back once CI is updated
    // stage('L2: ASR RNNT dev run') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('Speech to Text - RNNT') {
    //       steps {
    //         sh 'STRICT_NUMBA_COMPAT_CHECK=false python examples/asr/asr_transducer/speech_to_text_rnnt.py \
    //         --config-path="../conf/contextnet_rnnt/" --config-name="config_rnnt.yaml" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         model.train_ds.batch_size=2 \
    //         model.validation_ds.batch_size=2 \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_rnnt_results'
    //         sh 'rm -rf examples/asr/speech_to_text_rnnt_results'
    //       }
    //     }
    //     stage('L2: Speech to Text RNNT WPE') {
    //       steps {
    //         sh 'STRICT_NUMBA_COMPAT_CHECK=false python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    //         --config-path="../conf/contextnet_rnnt/" --config-name="config_rnnt_bpe.yaml" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         model.train_ds.batch_size=2 \
    //         model.validation_ds.batch_size=2 \
    //         model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
    //         model.tokenizer.type="wpe" \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_rnnt_wpe_results'
    //         sh 'rm -rf examples/asr/speech_to_text_rnnt_wpe_results'
    //       }
    //     }
    //     stage('L3: Speech to Text Hybrid Transducer-CTC WPE') {
    //       steps {
    //         sh 'STRICT_NUMBA_COMPAT_CHECK=false python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    //         --config-path="../conf/conformer/hybrid_transducer_ctc/conformer_hybrid_transducer_ctc/" --config-name="conformer_hybrid_transducer_ctc_bpe.yaml" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         model.encoder.n_layers= 2 \
    //         model.train_ds.batch_size=2 \
    //         model.validation_ds.batch_size=2 \
    //         model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
    //         model.tokenizer.type="wpe" \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_hybrid_transducer_ctc_wpe_results'
    //         sh 'rm -rf examples/asr/speech_to_text_hybrid_transducer_ctc_wpe_results'
    //       }
    //     }
    //   }
    // }

    // stage('L2: Hybrid ASR RNNT-CTC dev run') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('Speech to Text Hybrid Transducer-CTC WPE') {
    //       steps {
    //         sh 'STRICT_NUMBA_COMPAT_CHECK=false python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
    //         --config-path="../conf/conformer/hybrid_transducer_ctc/conformer_hybrid_transducer_ctc/" --config-name="conformer_hybrid_transducer_ctc_bpe.yaml" \
    //         model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
    //         model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
    //         model.encoder.n_layers= 2 \
    //         model.train_ds.batch_size=2 \
    //         model.validation_ds.batch_size=2 \
    //         model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
    //         model.tokenizer.type="wpe" \
    //         trainer.devices=[0] \
    //         trainer.accelerator="gpu" \
    //         +trainer.fast_dev_run=True \
    //         exp_manager.exp_dir=examples/asr/speech_to_text_hybrid_transducer_ctc_wpe_results'
    //         sh 'rm -rf examples/asr/speech_to_text_hybrid_transducer_ctc_wpe_results'
    //       }
    //     }
    //   }
    // }

    stage('L2: ASR Multi-dataloader dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text multi-dataloader') {
          steps {
            sh 'python examples/asr/asr_ctc/speech_to_text_ctc.py \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=[/home/TestData/an4_dataset/an4_val.json,/home/TestData/an4_dataset/an4_val.json] \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            +trainer.num_sanity_val_steps=1 \
            exp_manager.exp_dir=examples/asr/speech_to_text_results'
            sh 'rm -rf examples/asr/speech_to_text_results'
          }
        }

        stage('Speech to Label multi-dataloader') {
          steps {
            sh 'python examples/asr/speech_classification/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=[/home/TestData/speech_commands/test_manifest.json,/home/TestData/speech_commands/test_manifest.json] \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            +trainer.num_sanity_val_steps=1 \
            model.preprocessor._target_=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            ~model.preprocessor.window_size \
            ~model.preprocessor.window_stride \
            ~model.preprocessor.window \
            ~model.preprocessor.n_mels \
            ~model.preprocessor.n_mfcc \
            ~model.preprocessor.n_fft \
            exp_manager.exp_dir=examples/asr/speech_to_label_results'
            sh 'rm -rf examples/asr/speech_to_label_results'
          }
        }
      }
    }

    stage('L2: ASR Adapters') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Linear Adapters') {
          steps {
            sh 'python examples/asr/asr_adapters/train_asr_adapter.py \
            model.pretrained_model="stt_en_conformer_ctc_small" \
            model.adapter.adapter_name="an4" \
            model.adapter.linear.in_features=176 \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            trainer.max_steps=5 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_adapters_results'
            sh 'rm -rf examples/asr/speech_to_text_adapters_results'
          }
        }
        stage('RelPos MHA Adapters') {
          steps {
            sh 'python examples/asr/asr_adapters/train_asr_adapter.py \
            model.pretrained_model="stt_en_conformer_ctc_small" \
            model.adapter.adapter_name="encoder:an4" \
            model.adapter.adapter_type="tiny_attn" \
            model.adapter.tiny_attn.n_feat=176 \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            trainer.max_steps=5 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_adapters_mha_results'
            sh 'rm -rf examples/asr/speech_to_text_adapters_mha_results'
          }
        }

      }
    }

    stage('L2: Speech Transcription') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text Transcribe') {
          steps {
            sh 'python examples/asr/transcribe_speech.py \
            pretrained_name="QuartzNet15x5Base-En" \
            audio_dir="/home/TestData/an4_transcribe/test_subset/" \
            output_filename="stt_test_res.json" \
            amp=true'
            sh 'rm -rf stt_test_res.json'
          }
        }
      }
    }
    stage('L2: Transducer alignment') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Running pytest') {
          steps {
            sh 'pytest tests/collections/asr/decoding/rnnt_alignments_check.py --durations=-1'
          }
        }
      }
    }

    stage('L2: Segmentation Tool') {
      when {
            anyOf {
              branch 'main'
              changeRequest target: 'main'
            }
      }
      stages {
        stage('Install ctc_segmentation requirements') {
            steps {
            sh 'cd tools/ctc_segmentation && \
            pip install -r requirements.txt && \
            apt-get update && apt-get install libsox-fmt-all -y'
            }
        }

        stage('Parallel ctc_segmentation test') {
          failFast true
          parallel {
            stage('L2: Eng CitriNet with .wav') {
              steps {
                sh 'cd tools/ctc_segmentation && \
            TIME=`date +"%Y-%m-%d-%T"` && \
            /bin/bash run_segmentation.sh \
            --MODEL_NAME_OR_PATH="stt_en_citrinet_512_gamma_0_25" \
            --DATA_DIR=/home/TestData/ctc_segmentation/eng \
            --OUTPUT_DIR=/home/TestData/ctc_segmentation/eng/output${TIME} \
            --LANGUAGE=en \
            --USE_NEMO_NORMALIZATION="TRUE" && \
            python /home/TestData/ctc_segmentation/verify_alignment.py \
            -r /home/TestData/ctc_segmentation/eng/eng_valid_segments_1.7.txt \
            -g /home/TestData/ctc_segmentation/eng/output${TIME}/verified_segments/nv_test_segments.txt && \
            rm -rf /home/TestData/ctc_segmentation/eng/output${TIME}'
              }
            }
            stage('L2: Ru QN with mp3') {
              steps {
                sh 'cd tools/ctc_segmentation && \
            TIME=`date +"%Y-%m-%d-%T"` && \
            /bin/bash run_segmentation.sh \
            --MODEL_NAME_OR_PATH=/home/TestData/ctc_segmentation/QuartzNet15x5-Ru-e512-wer14.45.nemo \
            --DATA_DIR=/home/TestData/ctc_segmentation/ru \
            --OUTPUT_DIR=/home/TestData/ctc_segmentation/ru/output${TIME} \
            --LANGUAGE=ru \
            --ADDITIONAL_SPLIT_SYMBOLS=";" && \
            python /home/TestData/ctc_segmentation/verify_alignment.py \
            -r /home/TestData/ctc_segmentation/ru/valid_ru_segments_1.7.txt \
            -g /home/TestData/ctc_segmentation/ru/output${TIME}/verified_segments/ru_segments.txt && \
            rm -rf /home/TestData/ctc_segmentation/ru/output${TIME}'
              }
            }
          }
        }
      }
    }

    stage('L2: G2P Models') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('G2P Conformer training, evaluation and inference') {
          steps {
            sh 'cd examples/tts/g2p && \
                TIME=`date +"%Y-%m-%d-%T"` && OUTPUT_DIR_CONFORMER=output_ctc_${TIME} && \
                python g2p_train_and_evaluate.py \
                    train_manifest=/home/TestData/g2p/g2p.json \
                    validation_manifest=/home/TestData/g2p/g2p.json \
                    model.test_ds.manifest_filepath=/home/TestData/g2p/g2p.json \
                    model.tokenizer.dir=/home/TestData/g2p/tokenizer_spe_unigram_v512 \
                    trainer.max_epochs=1 \
                    model.max_source_len=64 \
                    trainer.devices=[0] \
                    do_training=True \
                    do_testing=True \
                    exp_manager.exp_dir=${OUTPUT_DIR_CONFORMER} \
                    +exp_manager.use_datetime_version=False\
                    +exp_manager.version=test \
                    --config-name=g2p_conformer_ctc && \
                python g2p_inference.py \
                    pretrained_model=${OUTPUT_DIR_CONFORMER}/G2P-Conformer-CTC/test/checkpoints/G2P-Conformer-CTC.nemo \
                    manifest_filepath=/home/TestData/g2p/g2p.json \
                    phoneme_field=text'
              }
            }
            // TODO: pleasefixme @redoctopus
            // stage('ByT5G2P training, evaluation and inference') {
            //   steps {
            //     sh 'cd examples/tts/g2p && \
            //         TIME=`date +"%Y-%m-%d-%T"` && OUTPUT_DIR_T5=output_byt5_${TIME} && \
            //         python g2p_train_and_evaluate.py \
            //             train_manifest=/home/TestData/g2p/g2p.json \
            //             validation_manifest=/home/TestData/g2p/g2p.json \
            //             model.test_ds.manifest_filepath=/home/TestData/g2p/g2p.json \
            //             trainer.max_epochs=1 \
            //             model.max_source_len=64 \
            //             trainer.devices=[1] \
            //             do_training=True \
            //             do_testing=True \
            //             exp_manager.exp_dir=${OUTPUT_DIR_T5} \
            //             +exp_manager.use_datetime_version=False\
            //             +exp_manager.version=test && \
            //         python g2p_inference.py \
            //             pretrained_model=${OUTPUT_DIR_T5}/T5G2P/test/checkpoints/T5G2P.nemo \
            //             manifest_filepath=/home/TestData/g2p/g2p.json \
            //             phoneme_field=text'
            //   }
            // }
           stage('HeteronymClassificationModel training, evaluation and inference') {
              steps {
                sh 'cd examples/tts/g2p && \
                    TIME=`date +"%Y-%m-%d-%T"` && OUTPUT_DIR=output_${TIME} && \
                    python g2p_heteronym_classification_train_and_evaluate.py \
                        train_manifest=/home/TestData/g2p/manifest.json \
                        validation_manifest=/home/TestData/g2p/manifest.json \
                        test_manifest=/home/TestData/g2p/manifest.json \
                        model.wordids=/home/TestData/g2p/wordids.tsv \
                        trainer.max_epochs=1 \
                        model.max_seq_length=64 \
                        do_training=True \
                        do_testing=True \
                        exp_manager.exp_dir=${OUTPUT_DIR} \
                        +exp_manager.use_datetime_version=False\
                        +exp_manager.version=test && \
                    python g2p_heteronym_classification_inference.py \
                        manifest=/home/TestData/g2p/manifest.json \
                        pretrained_model=${OUTPUT_DIR}/HeteronymClassification/test/checkpoints/HeteronymClassification.nemo \
                        output_manifest=preds.json'
              }
            }
          }
        }

    // TODO: add test once megatron-bert is supported again
    // stage('L2: Multi-GPU Megatron finetuning') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('L2: Cased Megatron finetuning on MRPC') {
    //       steps {
    //         sh 'cd examples/nlp/glue_benchmark && \
    //     python glue_benchmark.py \
    //     model.dataset.data_dir=/home/TestData/nlp/glue_fake/MRPC \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.use_cache=false \
    //     model.language_model.pretrained_model_name=megatron-bert-345m-cased \
    //     trainer.accelerator=gpu \
    //     trainer.strategy=ddp \
    //     exp_manager=null'
    //       }
    //     }
    //   }
    // }

    stage('L2: STS-b') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('GLUE STS-b with AlBERT') {
          steps {
            sh 'python examples/nlp/glue_benchmark/glue_benchmark.py \
            model.dataset.use_cache=false \
            model.task_name=sts-b \
            model.dataset.data_dir=/home/TestData/nlp/glue_fake/STS-B \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=True \
            model.language_model.pretrained_model_name=albert-base-v1 \
            exp_manager=null'
          }
        }
        stage('Test Restore Punctuation & Capitalization with AlBERT') {
          steps {
            sh 'data_dir="$(mktemp -d -p "$(pwd)")" && \
            cp /home/TestData/nlp/token_classification_punctuation/*.txt "${data_dir}"/ && \
            python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
              +do_training=false \
              +do_testing=true \
              pretrained_model=/home/TestData/nlp/pretrained_models/Punctuation_and_Capitalization_albert.nemo \
              +model.test_ds.use_cache=false \
              ~model.train_ds \
              ~model.validation_ds \
              model.test_ds.ds_item="${data_dir}" \
              trainer.devices=[1] \
              trainer.accelerator="gpu" \
              exp_manager=null && \
            rm -rf "${data_dir}"'
          }
        }
//         stage('Test Restore Punctuation & Capitalization with RoBERTa') {
//           steps {
//             sh 'data_dir="$(mktemp -d -p "$(pwd)")" && \
//             cp /home/TestData/nlp/token_classification_punctuation/*.txt "${data_dir}"/ && \
//             python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
//               +do_training=false \
//               +do_testing=true \
//               pretrained_model=/home/TestData/nlp/pretrained_models/Punctuation_and_Capitalization_roberta.nemo \
//               +model.test_ds.use_cache=false \
//               ~model.train_ds \
//               ~model.validation_ds \
//               model.test_ds.ds_item="${data_dir}" \
//               trainer.devices=[1] \
//               trainer.accelerator="gpu" \
//               exp_manager=null && \
//             rm -rf "${data_dir}"'
//           }
//         }
      }
    }
    stage('L2: Dialogue Classification') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Dialogue: Intent and slot classification using GPT') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            model.dataset.data_dir=/home/TestData/nlp/sgd_small \
            model.language_model.lm_checkpoint=/home/TestData/nlp/gpt2/pytorch_model.bin\
            model.tokenizer.vocab_file=/home/TestData/nlp/gpt2/vocab.json\
            model.dataset.dialogues_example_dir=sgd_gen_outputs \
            model.dataset.task_name=debug_sample \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[0] \
            model.dataset.use_cache=false \
            model.tokenizer.special_tokens={pad_token:"endoftext"} \
            model.tokenizer.tokenizer_name=gpt2 \
            model.tokenizer.vocab_file=/home/TestData/nlp/gpt2/vocab.json\
            model.language_model.pretrained_model_name=/home/TestData/nlp/gpt2 \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_gen_outputs'
          }
        }
        stage('Intent and slot classification using SGDQA') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            model.dataset.data_dir=/home/TestData/nlp/sgd_small \
            model.dataset.dialogues_example_dir=sgd_gen_bert_outputs \
            model.dataset.task_name=debug_sample \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.dataset.num_tasks=6 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[0] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-cased \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_gen_bert_outputs'
          }
        }
        stage('Intent and slot classification using IntentSlotClassificationModel') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            model.dataset.data_dir=/home/TestData/nlp/processed_assistant \
            model.dataset.dialogues_example_dir=sgd_gen_bert_intent_classification_outputs \
            model.dataset.task=assistant \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[0] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-uncased \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_gen_bert_intent_classification_outputs'
          }
        }
        stage('Intent classification using ZeroShotIntentModel') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            do_training=False \
            model.dataset.data_dir=/home/TestData/nlp/drive_thru_revised \
            model.original_nemo_checkpoint=/home/TestData/nlp/drive_thru_revised/zeroshotintent_en_bert_base_uncased.nemo \
            model.dataset.dialogues_example_dir=sgd_gen_zero_shot_intent_classification_outputs \
            model.dataset.task=zero_shot \
            model.dataset.prompt_template="This example is" \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[1] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-uncased \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_gen_zero_shot_intent_classification_outputs'
          }
        }
        stage('Design Intent classification using ZeroShotIntentModel') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            do_training=False \
            model.dataset.data_dir=/home/TestData/nlp/design_dataset \
            model.original_nemo_checkpoint=/home/TestData/nlp/drive_thru_revised/zeroshotintent_en_bert_base_uncased.nemo \
            model.dataset.dialogues_example_dir=design_zero_shot_intent_classification_outputs \
            model.dataset.task=design \
            model.dataset.prompt_template="This example is related to" \
            model.library=megatron \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[1] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-uncased \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf design_zero_shot_intent_classification_outputs'
          }
        }
        stage('Design Intent classification using ZeroShotIntentModel BART Classifier') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            do_training=False \
            model.dataset.data_dir=/home/TestData/nlp/design_dataset \
            model.original_nemo_checkpoint=/home/TestData/nlp/drive_thru_revised/zeroshotintent_en_bert_base_uncased.nemo \
            model.dataset.dialogues_example_dir=design_zero_shot_intent_classification_bart_outputs \
            model.dataset.task=design \
            model.dataset.prompt_template="This example is related to" \
            model.library=huggingface \
            trainer.devices=[1] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-uncased \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf design_zero_shot_intent_classification_bart_outputs'
          }
        }
        stage('Design Intent classification using DialogueNearestNeighbourModel') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            do_training=False \
            model.dataset.data_dir=/home/TestData/nlp/design_dataset \
            model.dataset.dialogues_example_dir=design_dialogue_nearest_neighbour_classification_outputs \
            model.dataset.task=design \
            model.dataset.prompt_template="" \
            model.library=huggingface \
            trainer.devices=[0] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=sentence-transformers/all-MiniLM-L6-v2 \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf design_dialogue_nearest_neighbour_classification_outputs'
          }
        }
      }
    }
    stage('L2: Dialogue Generation') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Dialogue: Answer Extender using DialogueS2SGenerationModel') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            do_training=False \
            model.dataset.data_dir=/home/TestData/nlp/ms-marco-qa \
            model.dataset.dialogues_example_dir=answer_extender_s2s \
            model.dataset.task=ms_marco \
            model.library=huggingface \
            model.dataset.debug_mode=True \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[1] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=facebook/bart-large \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf answer_extender_s2s'
          }
        }
        stage('Dialogue: SGD Based Answer Extender using DialogueS2SGenerationModel') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            do_training=False \
            model.dataset.data_dir=/home/TestData/nlp/sgd_small \
            model.dataset.dialogues_example_dir=sgd_answer_extender_s2s \
            model.dataset.task_name=debug_sample \
            model.dataset.task=sgd_generation \
            model.dataset.input_field=utterance+system_actions \
            model.dataset.output_field=system_utterance \
            model.dataset.use_cache=false \
            model.dataset.system_utterance=next_turn \
            model.dataset.debug_mode=True \
            model.dataset.prompt_template=slots_values \
            model.library=huggingface \
            trainer.max_steps=1 \
            trainer.max_epochs=1 \
            model.train_ds.batch_size=2 \
            model.validation_ds.batch_size=2 \
            model.test_ds.batch_size=2 \
            model.nemo_path=null \
            trainer.val_check_interval=0.0 \
            trainer.devices=[0] \
            model.language_model.pretrained_model_name=facebook/bart-large \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf sgd_answer_extender_s2s'
          }
        }
      }
    }
//     stage('L2: Dialogue Generation Part 2') {
//       when {
//         anyOf {
//           branch 'main'
//           changeRequest target: 'main'
//         }
//       }
//       failFast true
//       parallel {
//         stage('Dialogue: Answer Extender using DialogueGPTGenerationModel') {
//           steps {
//             sh 'cd examples/nlp/dialogue && \
//             python dialogue.py \
//             do_training=False \
//             model.dataset.data_dir=/home/TestData/nlp/ms-marco-qa \
//             model.dataset.dialogues_example_dir=answer_extender \
//             model.library=huggingface \
//             model.dataset.task=ms_marco \
//             model.dataset.debug_mode=True \
//             trainer.val_check_interval=0.0 \
//             trainer.devices=[0] \
//             model.dataset.use_cache=false \
//             model.language_model.pretrained_model_name=gpt2 \
//             trainer.accelerator=gpu \
//             exp_manager=null  && \
//             rm -rf answer_extender'
//           }
//         }
//       }
//     }
    stage('L2: COPY') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Dialogue: Answer Extender using DialogueGPTGenerationModel') {
          steps {
            sh 'cd examples/nlp/dialogue && \
            python dialogue.py \
            do_training=False \
            model.dataset.data_dir=/home/TestData/nlp/ms-marco-qa \
            model.dataset.dialogues_example_dir=answer_extender \
            model.library=huggingface \
            model.dataset.task=ms_marco \
            model.dataset.debug_mode=True \
            trainer.val_check_interval=0.0 \
            trainer.devices=[0] \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name=gpt2 \
            trainer.accelerator=gpu \
            exp_manager=null  && \
            rm -rf answer_extender'
          }
        }
      }
    }
    stage('L2: Duplex Text Normalization') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Duplex Text Normalization with Tarred dataset') {
          steps {
            sh 'cd examples/nlp/duplex_text_normalization && \
            python duplex_text_normalization_train.py \
            data.validation_ds.data_path=/home/TestData/nlp/duplex_text_norm/small_test.tsv \
            mode=tn \
            lang=en \
            tagger_model.do_training=false \
            decoder_model.transformer=t5-small \
            data.validation_ds.batch_size=2 \
            data.train_ds.use_cache=false \
            data.validation_ds.use_cache=false \
            data.test_ds.batch_size=2 \
            data.train_ds.decoder_data_augmentation=false \
            data.train_ds.num_workers=2 \
            decoder_trainer.devices=[0,1] \
            decoder_trainer.accelerator="gpu" \
            data.train_ds.use_tarred_dataset=true \
            +decoder_trainer.fast_dev_run=true \
            decoder_exp_manager.create_checkpoint_callback=false \
            data.train_ds.tar_metadata_file=/home/TestData/nlp/duplex_text_norm/tarred_small/metadata.json \
            data.test_ds.use_cache=false \
            data.test_ds.data_path=/home/TestData/nlp/duplex_text_norm/small_test.tsv'
          }
        }
      }
    }
    // Runs out of memory on the 12G TITAN V (GPU 0 on main CI)
    // TODO: add when megatron bert is supported again in NeMo
    // stage('L2: MegaBERT Token Classification') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh 'cd examples/nlp/token_classification && \
    //     python token_classification_train.py \
    //     model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
    //     model.language_model.pretrained_model_name=megatron-bert-345m-uncased \
    //     model.train_ds.batch_size=10 \
    //     model.dataset.max_seq_length=50 \
    //     model.dataset.use_cache=false \
    //     trainer.accelerator=gpu \
    //     trainer.strategy=ddp \
    //     trainer.precision=16 \
    //     trainer.devices=[1] \
    //     trainer.accelerator="gpu" \
    //     +trainer.fast_dev_run=true \
    //     exp_manager=null'
    //   }
    // }

    stage('L2: BERT Text Classification') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('Text Classification with BERT Test') {
          steps {
            sh 'cd examples/nlp/text_classification && \
            python text_classification_with_bert.py \
            model.dataset.num_classes=6 \
            model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
            model.validation_ds.file_path=/home/TestData/nlp/retail_text_classification/dev.tsv \
            model.language_model.pretrained_model_name=distilbert-base-uncased \
            model.train_ds.batch_size=10 \
            model.dataset.max_seq_length=50 \
            model.dataset.use_cache=false \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            exp_manager=null'
          }
        }
      }
    }

    stage('L2: Parallel BERT Question-Answering SQUAD v1.1 & v2.0') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('BERT SQUAD 1.1') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.test_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            model.test_ds.num_samples=2 \
            model.test_ds.batch_size=2 \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
        stage('BERT SQUAD 2.0') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
      }
    }

    stage('L2: Parallel BART Question-Answering SQUAD v1.1 & v2.0') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('BART SQUAD 1.1') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.dataset.check_if_answer_in_context=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.test_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            model.test_ds.num_samples=2 \
            model.test_ds.batch_size=2 \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            model.language_model.pretrained_model_name=facebook/bart-base \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
        stage('BART SQUAD 2.0') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.dataset.check_if_answer_in_context=false \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=facebook/bart-base \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
      }
    }

    stage('L2: Parallel GPT2 Question-Answering SQUAD v1.1 & v2.0') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('GPT2 SQUAD 1.1') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.dataset.check_if_answer_in_context=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.test_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            model.test_ds.num_samples=2 \
            model.test_ds.batch_size=2 \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            model.language_model.pretrained_model_name=gpt2 \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
        stage('GPT2 SQUAD 2.0') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.dataset.check_if_answer_in_context=false \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            trainer.max_epochs=1 \
            trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=gpt2 \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.devices=[1] \
            trainer.accelerator="gpu" \
            exp_manager=null'
          }
        }
      }
    }

    stage('L2: Intent and Slot Classification Tasks') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Intent and Slot Classification') {
          steps {
            sh 'cd examples/nlp/intent_slot_classification && \
            python intent_slot_classification.py \
            model.data_dir=/home/TestData/nlp/retail \
            model.validation_ds.prefix=dev \
            model.test_ds.prefix=dev \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=checkpoints'
            sh 'rm -rf checkpoints'
          }
        }
        stage('L2: Multi-Label Intent and Slot Classification') {
          steps {
            sh 'cd examples/nlp/intent_slot_classification && \
            python multi_label_intent_slot_classification.py \
            model.data_dir=/home/TestData/nlp/new_multiatis \
            model.validation_ds.prefix=dev \
            model.test_ds.prefix=dev \
            trainer.devices=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=checkpoints2'
            sh 'rm -rf checkpoints2'
          }
        }
      }
    }

    // TODO: add when megatron-bert is supported again
    // stage('L2: Model Parallel Size 2 Megatron Text Classification') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/text_classification && \
    //     python text_classification_with_bert.py \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     trainer.num_nodes=1 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.num_classes=6 \
    //     model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
    //     model.train_ds.batch_size=4 \
    //     model.language_model.pretrained_model_name=megatron-bert-uncased \
    //     model.language_model.config_file=/home/TestData/nlp/mp_2_bert_toy/config.json \
    //     model.language_model.lm_checkpoint=/home/TestData/nlp/mp_2_bert_toy/iter_2000000 \
    //     model.nemo_path=null \
    //     ~model.infer_samples \
    //     exp_manager=null'
    //   }
    // }

    // stage('L2: Model Parallel Size 2 Megatron Autoresume') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/text_classification && \
    //     python text_classification_with_bert.py \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     trainer.num_nodes=1 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     trainer.max_epochs=1 \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.num_classes=6 \
    //     model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
    //     model.train_ds.batch_size=4 \
    //     model.language_model.pretrained_model_name=megatron-bert-uncased \
    //     model.language_model.config_file=/home/TestData/nlp/mp_2_bert_toy/config.json \
    //     model.language_model.lm_checkpoint=/home/TestData/nlp/mp_2_bert_toy/iter_2000000 \
    //     model.nemo_path=null \
    //     ~model.infer_samples \
    //     +exp_manager.explicit_log_dir=/home/TestData/nlp/mp_autoresume \
    //     +exp_manager.resume_if_exists=true'
    //   }
    // }

    // stage('L2: Model Parallel Size 2 Megatron Evaluation from .nemo') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/text_classification && \
    //     python model_parallel_text_classification_evaluation.py \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     trainer.num_nodes=1 \
    //     model.dataset.num_classes=6 \
    //     model.test_ds.file_path=/home/TestData/nlp/retail_text_classification/dev.tsv \
    //     model.nemo_path=/home/TestData/nlp/mp_2_nemo/retail_text_class_350M.nemo \
    //     exp_manager=null'
    //   }
    // }

    // stage('L2: Model Parallel Size 2 Megatron Train from .nemo') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/token_classification && \
    //     python token_classification_train.py \
    //     pretrained_model=/home/TestData/nlp/mp_2_nemo/ner_350M.nemo \
    //     model.dataset.data_dir=/home/TestData/nlp/ner/ \
    //     model.train_ds.batch_size=2 \
    //     model.dataset.use_cache=false \
    //     trainer.devices=[0,1] \
    //     trainer.accelerator="gpu" \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.class_balancing="weighted_loss" \
    //     exp_manager=null'
    //   }
    // }

    stage('L2: Parallel NLP Examples 2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('NER finetuning from pretrained Test') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python token_classification_train.py \
            pretrained_model=ner_en_bert \
            model.dataset.data_dir=/home/TestData/nlp/ner/ \
            model.train_ds.batch_size=2 \
            model.dataset.use_cache=false \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            model.dataset.class_balancing="weighted_loss" \
            exp_manager.exp_dir=null'
          }
        }
        stage ('Punctuation and capitalization finetuning from pretrained test') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            data_dir="$(mktemp -d -p "$(pwd)")" && \
            cp /home/TestData/nlp/token_classification_punctuation/*.txt "${data_dir}"/ && \
            python punctuation_capitalization_train_evaluate.py \
              pretrained_model=punctuation_en_bert \
              model.train_ds.ds_item="${data_dir}" \
              model.validation_ds.ds_item="${data_dir}" \
              model.test_ds.ds_item="${data_dir}" \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[1] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              exp_manager.exp_dir=null && \
            rm -rf "${data_dir}"'
          }
        }
        stage ('NER with TurkuNLP/bert-base-finnish-cased-v1') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python token_classification_train.py \
            model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.fast_dev_run=true \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name="TurkuNLP/bert-base-finnish-cased-v1" \
            exp_manager.exp_dir=null'
          }
        }
        stage('Evaluation script for Token Classification') {
          steps {
            sh 'python examples/nlp/token_classification/token_classification_evaluate.py \
            model.dataset.data_dir=/home/TestData/nlp/ner/ \
            model.dataset.use_cache=false \
            pretrained_model=/home/TestData/nlp/pretrained_models/NER_Model_with_BERT_base_uncased.nemo'
          }
        }
        stage('Evaluation script for Punctuation') {
          steps {
            sh 'data_dir="$(mktemp -d -p "$(pwd)")" && \
            cp /home/TestData/nlp/token_classification_punctuation/*.txt "${data_dir}"/ && \
            python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
              +do_training=false \
              +do_testing=true \
              model.test_ds.ds_item="${data_dir}" \
              ~model.train_ds \
              ~model.validation_ds \
              +model.test_ds.use_cache=false \
              pretrained_model=/home/TestData/nlp/pretrained_models/Punctuation_Capitalization_with_DistilBERT_base_uncased.nemo && \
            rm -rf "${data_dir}"'
          }
        }
        stage('L2: Punctuation & Capitalization, 2GPUs with DistilBERT, Fine-tuning on different data') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            output_dir="$(mktemp -d -p "$(pwd)")" && \
            tmp_data_dir="$(mktemp -d -p "$(pwd)")" && \
            cp /home/TestData/nlp/token_classification_punctuation/*.txt "${tmp_data_dir}"/ && \
            python punctuation_capitalization_train_evaluate.py \
              model.train_ds.use_tarred_dataset=false \
              model.train_ds.ds_item="${tmp_data_dir}" \
              model.validation_ds.ds_item="${tmp_data_dir}" \
              model.test_ds.ds_item="${tmp_data_dir}" \
              model.language_model.pretrained_model_name=distilbert-base-uncased \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.accelerator="gpu" \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              +exp_manager.explicit_log_dir="${output_dir}" \
              +do_testing=true && \
            tmp_data_dir_2="$(mktemp -d -p "$(pwd)")" && \
            mv "${tmp_data_dir}"/* "${tmp_data_dir_2}" && \
            rm -rf "${tmp_data_dir}" && \
            python punctuation_capitalization_train_evaluate.py \
              model.train_ds.use_tarred_dataset=false \
              model.train_ds.ds_item="${tmp_data_dir_2}" \
              model.validation_ds.ds_item="${tmp_data_dir_2}" \
              model.test_ds.ds_item="${tmp_data_dir_2}" \
              pretrained_model="${output_dir}/checkpoints/Punctuation_and_Capitalization.nemo" \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.accelerator="gpu" \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              exp_manager=null && \
            rm -rf /workspace/NeMo/examples/nlp/token_classification/nemo_experiments \
              "${tmp_data_dir_2}" \
              "${output_dir}"'
          }
        }
      }
    }
    
    stage('Punctuation & Capitalization tarred dataset') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      stages {
        stage('create and use tarred dataset') {
          steps {
            sh 'data_dir="$(mktemp -d -p "$(pwd)")" && \
            cp -r /home/TestData/nlp/token_classification_punctuation/*.txt \
              /home/TestData/nlp/token_classification_punctuation/wmt_wiki_10000 \
              "${data_dir}"/ && \
            usual_data=${data_dir}/wmt_wiki_10000 && \
            output_dir="$(mktemp -d -p "$(pwd)")" && \
            tarred_data=${output_dir}/train_tarred && \
            tokens_in_batch=2000 && \
            max_seq_length=512 && \
            lm_model=distilbert-base-uncased && \
            python examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py \
              --text ${usual_data}/input.txt \
              --labels ${usual_data}/labels.txt \
              --output_dir ${tarred_data} \
              --tokens_in_batch ${tokens_in_batch} \
              --max_seq_length 512 \
              --lines_per_dataset_fragment 2000 \
              --num_batches_per_tarfile 5 \
              --tar_file_prefix punctuation_capitalization \
              --tokenizer_name ${lm_model} \
              --use_fast_tokenizer \
              --pad_label O \
              --n_jobs 3 && \
            echo "Number of tarred files in dataset:" && \
            ls ${tarred_data}/*.tar | wc -l && \
            echo "Label id files in dataset:" && \
            ls ${tarred_data}/*.csv && \
            metadata_file=${tarred_data}/metadata.punctuation_capitalization.tokens${tokens_in_batch}.max_seq_length${max_seq_length}.${lm_model}.json && \
            python examples/nlp/token_classification/punctuation_capitalization_train_evaluate.py \
              model.validation_ds.ds_item="${data_dir}" \
              model.test_ds.ds_item="${data_dir}" \
              model.train_ds.ds_item=${tarred_data} \
              model.language_model.pretrained_model_name=${lm_model} \
              model.train_ds.use_tarred_dataset=true \
              model.train_ds.tar_metadata_file=${metadata_file} \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.accelerator="gpu" \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              +exp_manager.explicit_log_dir=${output_dir}/output && \
            rm -rf "${output_dir}" "${data_dir}"'
          }
        }
      }
    }
    
    stage('Punctuation & Capitalization, Different ways of passing labels to model') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      stages {
        stage('Punctuation & Capitalization, Using model.common_datasest_parameters.label_vocab_dir') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            work_dir="$(mktemp -d -p "$(pwd)")" && \
            label_vocab_dir="${work_dir}/labels" && \
            mkdir -p ${label_vocab_dir} && \
            data_dir="${work_dir}/data" && \
            mkdir -p "${data_dir}" && \
            cp /home/TestData/nlp/token_classification_punctuation/*.txt "${data_dir}" && \
            output_dir="${work_dir}/output" && \
            mkdir -p "${output_dir}" && \
            punct_label_vocab="${label_vocab_dir}/punct_label_vocab.csv" && \
            capit_label_vocab="${label_vocab_dir}/capit_label_vocab.csv" && \
            printf "O\n,\n.\n?\n" > "${punct_label_vocab}" && \
            printf "O\nU\n" > "${capit_label_vocab}" && \
            python punctuation_capitalization_train_evaluate.py \
              model.train_ds.use_tarred_dataset=false \
              model.train_ds.ds_item="${data_dir}" \
              model.validation_ds.ds_item="${data_dir}" \
              model.test_ds.ds_item="${data_dir}" \
              model.language_model.pretrained_model_name=distilbert-base-uncased \
              model.common_dataset_parameters.label_vocab_dir="${label_vocab_dir}" \
              model.class_labels.punct_labels_file="$(basename "${punct_label_vocab}")" \
              model.class_labels.capit_labels_file="$(basename "${capit_label_vocab}")" \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              +exp_manager.explicit_log_dir="${output_dir}" \
              +do_testing=false && \
            python punctuation_capitalization_train_evaluate.py \
              +do_training=false \
              +do_testing=true \
              ~model.train_ds \
              ~model.validation_ds \
              model.test_ds.ds_item="${data_dir}" \
              pretrained_model="${output_dir}/checkpoints/Punctuation_and_Capitalization.nemo" \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              exp_manager=null && \
            rm -rf "${work_dir}"'
          }
        }
        stage('Punctuation & Capitalization, Using model.common_datasest_parameters.{punct,capit}_label_ids') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            work_dir="$(mktemp -d -p "$(pwd)")" && \
            output_dir="${work_dir}/output" && \
            mkdir -p "${output_dir}" && \
            data_dir="${work_dir}/data" && \
            mkdir -p "${data_dir}" && \
            cp /home/TestData/nlp/token_classification_punctuation/*.txt "${data_dir}" && \
            conf_name=punctuation_capitalization_config_with_ids && \
            cp conf/punctuation_capitalization_config.yaml "${work_dir}/${conf_name}.yaml" && \
            sed -i $\'s/punct_label_ids: null/punct_label_ids: {O: 0, \\\',\\\': 1, .: 2, \\\'?\\\': 3}/\' \
              "${work_dir}/${conf_name}.yaml" && \
            sed -i $\'s/capit_label_ids: null/capit_label_ids: {O: 0, U: 1}/\' \
              "${work_dir}/${conf_name}.yaml" && \
            python punctuation_capitalization_train_evaluate.py \
              --config-path "${work_dir}" \
              --config-name "${conf_name}" \
              model.train_ds.use_tarred_dataset=false \
              model.train_ds.ds_item="${data_dir}" \
              model.validation_ds.ds_item="${data_dir}" \
              model.test_ds.ds_item="${data_dir}" \
              model.language_model.pretrained_model_name=distilbert-base-uncased \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              +exp_manager.explicit_log_dir="${output_dir}" \
              +do_testing=false && \
            python punctuation_capitalization_train_evaluate.py \
              +do_training=false \
              +do_testing=true \
              ~model.train_ds \
              ~model.validation_ds \
              model.test_ds.ds_item="${data_dir}" \
              pretrained_model="${output_dir}/checkpoints/Punctuation_and_Capitalization.nemo" \
              +model.train_ds.use_cache=false \
              +model.validation_ds.use_cache=false \
              +model.test_ds.use_cache=false \
              trainer.devices=[0,1] \
              trainer.strategy=ddp \
              trainer.max_epochs=1 \
              exp_manager=null && \
            rm -rf "${work_dir}"'
          }
        }
      }
    }
    stage('Punctuation & Capitalization inference') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      stages {
        stage('Restore punctuation and capitalization in long text') {
          steps {
            sh 'output_dir="$(mktemp -d -p "$(pwd)")" && \
            python examples/nlp/token_classification/punctuate_capitalize_infer.py \
              --input_manifest /home/TestData/nlp/token_classification_punctuation/iwslt_tst2019.manifest \
              --output_text "${output_dir}/iwslt_inference_result.txt" \
              --max_seq_length 92 \
              --step 8 \
              --margin 16 \
              --pretrained_name punctuation_en_bert \
              --batch_size 32 && \
            rm -rf "${output_dir}"'
          }
        }
      }
    }

    stage('L2: Parallel Pretraining BERT pretraining from Text/Preprocessed') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Pretraining BERT pretraining from Text') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_text_config.yaml \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              trainer.precision=16 \
              +trainer.fast_dev_run=true \
              model.train_ds.data_file=/home/TestData/nlp/wikitext-2/train.txt  \
              model.train_ds.batch_size=32 \
              model.validation_ds.data_file=/home/TestData/nlp/wikitext-2/valid.txt  \
              model.validation_ds.batch_size=32 \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/bert_3200.json \
              model.optim.lr=0.01 \
              model.optim.sched.warmup_ratio=0.1 \
              model.tokenizer.tokenizer_name=sentencepiece \
              model.tokenizer.tokenizer_model=/home/TestData/nlp/wikitext-2/tokenizer_bpe_v3193/tokenizer.model \
              model.mask_prob=0.15 \
              model.short_seq_prob=0.1 \
              exp_manager.exp_dir=PretrainingBERTFromText \
              '
              sh 'rm -f /home/TestData/nlp/wikitext-2/*.pkl'
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromText'
              sh 'ls -lha examples/nlp/language_modeling'
            }
        }
        stage('L2: Pretraining BERT from Preprocessed') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_preprocessed_config.yaml \
              trainer.devices=[1] \
              trainer.accelerator="gpu" \
              trainer.precision=16 \
              +trainer.fast_dev_run=false \
              +trainer.max_epochs=1 \
              +trainer.limit_val_batches=0 \
              +trainer.limit_train_batches=1 \
              model.train_ds.data_file=/home/TestData/nlp/wiki_book_mini/training \
              model.train_ds.batch_size=8 \
              model.language_model.lm_checkpoint=/home/TestData/nlp/bert_ckpts/nemo1.0/bert_base_uncased_mlm_final_1074591_nemo1.0.pt \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/uncased_L-12_H-768_A-12.json \
              model.optim.lr=0.875e-4 \
              model.optim.weight_decay=0.01 \
              model.optim.sched.warmup_ratio=0.01 \
              exp_manager.exp_dir=PretrainingBERTFromPreprocessed \
              exp_manager.create_checkpoint_callback=False \
              '
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromPreprocessed'
              sh 'ls -lha examples/nlp/language_modeling'
            }
        }
      }
    }

    stage('L2: Entity Linking') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('Self Alignment Pretraining BERT') {
           steps {
             sh 'cd examples/nlp/entity_linking && \
             python self_alignment_pretraining.py \
             project_dir=. \
             trainer.val_check_interval=3 \
             model.raw_data=None \
             model.train_ds.data_file=/home/TestData/nlp/entity_linking/tiny_example_train_pairs.tsv \
             model.validation_ds.data_file=/home/TestData/nlp/entity_linking/tiny_example_validation_pairs.tsv \
             model.train_ds.batch_size=8 \
             model.validation_ds.batch_size=8 \
             exp_manager.exp_dir=null'
          }
        }
      }
    }

    // TODO: remove +model.optim.capturable=True when Pytorch fix: https://github.com/pytorch/pytorch/pull/81858
    // is in the release container
    stage('L2: NMT Attention is All You Need Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Training Post-LN') {
            steps {
              sh 'python examples/nlp/machine_translation/enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=false \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              model.encoder.num_layers=1 \
              model.encoder.hidden_size=64 \
              model.encoder.inner_size=256 \
              model.decoder.num_layers=1 \
              model.decoder.hidden_size=64 \
              model.decoder.inner_size=256 \
              +model.optim.capturable=True \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.val_check_interval=2 \
              +trainer.limit_val_batches=1 \
              +trainer.max_steps=2 \
              trainer.precision=16 \
              +exp_manager.explicit_log_dir=examples/nlp/machine_translation/nmt_results \
              +exp_manager.create_checkpoint_callback=true \
              '
              sh 'python examples/nlp/machine_translation/enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              model.encoder.num_layers=1 \
              model.encoder.hidden_size=64 \
              model.encoder.inner_size=256 \
              model.decoder.num_layers=1 \
              model.decoder.hidden_size=64 \
              model.decoder.inner_size=256 \
              +model.optim.capturable=True \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.val_check_interval=10 \
              +trainer.limit_val_batches=1 \
              +trainer.limit_test_batches=1 \
              +trainer.max_steps=10 \
              +exp_manager.explicit_log_dir=examples/nlp/machine_translation/nmt_results \
              +exp_manager.create_checkpoint_callback=true \
              +exp_manager.resume_if_exists=True \
              '
              sh 'rm -rf examples/nlp/machine_translation/nmt_results'
            }
        }

        stage('L2: NMT Training Pre-LN') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              model.encoder.pre_ln=true \
              model.decoder.pre_ln=true \
              trainer.devices=[1] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              +trainer.limit_test_batches=2 \
              exp_manager=null \
              '
            }
        }
        stage('L2: NMT Multi-Validation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_testing=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref \
              model.validation_ds.src_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src] \
              model.validation_ds.tgt_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref] \
              model.test_ds.src_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src] \
              model.test_ds.tgt_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref] \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              +trainer.limit_test_batches=2 \
              exp_manager=null \
              '
            }
        }
      }
    }

    stage('L2: NMT Attention is All You Need Inference') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh 'cd examples/nlp/machine_translation && \
        python nmt_transformer_infer.py \
        --model=/home/TestData/nlp/nmt/toy_data/enes_v16k_s100k_6x6.nemo \
        --srctext=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.test.src \
        --tgtout=/home/TestData/nlp/nmt/toy_data/out.txt \
        --target_lang en \
        --source_lang de \
        '
      }
    }

    stage('L2: NMT Attention is All You Need Finetuning') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "cd examples/nlp/machine_translation && \
        python enc_dec_nmt_finetune.py \
        model_path=/home/TestData/nlp/nmt/toy_data/enes_v16k_s100k_6x6.nemo \
        trainer.devices=[0] \
        ~trainer.max_epochs \
        model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
        model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
        +trainer.val_check_interval=10 \
        +trainer.limit_val_batches=1 \
        +trainer.limit_test_batches=1 \
        +trainer.max_steps=10 \
        +exp_manager.exp_dir=examples/nlp/machine_translation/nmt_finetune \
        +exp_manager.create_checkpoint_callback=True \
        +exp_manager.checkpoint_callback_params.monitor=val_sacreBLEU \
        +exp_manager.checkpoint_callback_params.mode=max \
        +exp_manager.checkpoint_callback_params.save_best_model=true \
        "
        sh "rm -rf examples/nlp/machine_translation/nmt_finetune"
      }
    }


    stage('L2: NMT Tarred Dataset Creation') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Auto Tarred Dataset Creation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_training=false \
              model.preproc_out_dir=$PWD/preproc_out_dir \
              model.train_ds.use_tarred_dataset=true \
              model.train_ds.n_preproc_jobs=2 \
              model.train_ds.lines_per_dataset_fragment=500 \
              model.train_ds.num_batches_per_tarfile=10 \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.vocab_size=2000 \
              model.decoder_tokenizer.vocab_size=2000 \
              ~model.test_ds \
              trainer.devices=[0] \
              trainer.accelerator="gpu" \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }

        stage('L2: NMT Script Tarred Dataset Creation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python create_tarred_parallel_dataset.py \
              --src_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              --tgt_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              --out_dir $PWD/out_dir \
              --encoder_tokenizer_vocab_size=2000 \
              --decoder_tokenizer_vocab_size=2000 \
              --tokens_in_batch=1000 \
              --lines_per_dataset_fragment=500 \
              --num_batches_per_tarfile=10 \
              --n_preproc_jobs=2 \
              '
            }
        }
      }
    }
    // stage('L2: Megatron NMT Training TP=2') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh "python examples/nlp/machine_translation/megatron_nmt_training.py \
    //     trainer.devices=2 \
    //     trainer.accelerator=gpu \
    //     trainer.log_every_n_steps=1 \
    //     trainer.val_check_interval=10 \
    //     +trainer.limit_val_batches=2 \
    //     trainer.accumulate_grad_batches=1 \
    //     trainer.max_steps=10 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     exp_manager.exp_dir=examples/nlp/machine_translation/megatron_nmt_results \
    //     model.tensor_model_parallel_size=2 \
    //     model.seq_length=128 \
    //     model.encoder.num_layers=4 \
    //     model.encoder.hidden_size=64 \
    //     model.encoder.num_attention_heads=8 \
    //     model.encoder.activation='swiglu' \
    //     model.encoder.masked_softmax_fusion=False \
    //     model.encoder.bias_activation_fusion=False \
    //     model.encoder.activations_checkpoint_method='block' \
    //     model.encoder.activations_checkpoint_num_layers=1 \
    //     model.decoder.num_layers=2 \
    //     model.decoder.hidden_size=64 \
    //     model.decoder.num_attention_heads=8 \
    //     model.decoder.activation='swiglu' \
    //     model.decoder.masked_softmax_fusion=False \
    //     model.decoder.bias_activation_fusion=False \
    //     model.decoder.activations_checkpoint_method='block' \
    //     model.decoder.activations_checkpoint_num_layers=1 \
    //     model.micro_batch_size=2 \
    //     model.global_batch_size=4 \
    //     model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //     model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //     model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //     model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //     model.train_ds.num_workers=1 \
    //     model.validation_ds.num_workers=1 \
    //     ~model.test_ds \
    //     model.train_ds.dataset_type=text_memmap \
    //     model.encoder_tokenizer.library=sentencepiece \
    //     model.encoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
    //     model.decoder_tokenizer.library=sentencepiece \
    //     model.decoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model"
    //     // Change val_check_interval to 1 for resume as the len(dataloder) is 1 due to max_steps being the same as that of training and Lightning 2.0 raises an error
    //     // if val_check_interval > len(dataloder: https://github.com/Lightning-AI/lightning/blob/2.0.6/src/lightning/pytorch/loops/fit_loop.py#L259 at the beginning of fit_loop.run()
    //     sh "python examples/nlp/machine_translation/megatron_nmt_training.py \
    //     trainer.devices=2 \
    //     trainer.accelerator=gpu \
    //     trainer.log_every_n_steps=1 \
    //     trainer.val_check_interval=1 \
    //     +trainer.limit_val_batches=2 \
    //     trainer.accumulate_grad_batches=1 \
    //     trainer.max_steps=10 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     exp_manager.exp_dir=examples/nlp/machine_translation/megatron_nmt_results \
    //     model.tensor_model_parallel_size=2 \
    //     model.seq_length=128 \
    //     model.encoder.num_layers=4 \
    //     model.encoder.hidden_size=64 \
    //     model.encoder.num_attention_heads=8 \
    //     model.encoder.activation='swiglu' \
    //     model.encoder.masked_softmax_fusion=False \
    //     model.encoder.bias_activation_fusion=False \
    //     model.encoder.activations_checkpoint_method='block' \
    //     model.encoder.activations_checkpoint_num_layers=1 \
    //     model.decoder.num_layers=2 \
    //     model.decoder.hidden_size=64 \
    //     model.decoder.num_attention_heads=8 \
    //     model.decoder.activation='swiglu' \
    //     model.decoder.masked_softmax_fusion=False \
    //     model.decoder.bias_activation_fusion=False \
    //     model.decoder.activations_checkpoint_method='block' \
    //     model.decoder.activations_checkpoint_num_layers=1 \
    //     model.micro_batch_size=2 \
    //     model.global_batch_size=4 \
    //     model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //     model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //     model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //     model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //     model.train_ds.num_workers=1 \
    //     model.validation_ds.num_workers=1 \
    //     ~model.test_ds \
    //     model.train_ds.dataset_type=text_memmap \
    //     model.encoder_tokenizer.library=sentencepiece \
    //     model.encoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
    //     model.decoder_tokenizer.library=sentencepiece \
    //     model.decoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model"
    //     sh "rm -rf examples/nlp/machine_translation/megatron_nmt_results"
    //   }
    // }
    stage('L2: Megatron BART Perceiver MIM Training TP=2') {
      // Testing Megatron hidden transformations
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/megatron_mim_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.arch=perceiver \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.micro_batch_size=2 \
        model.global_batch_size=4 \
        model.data.data_impl=text_mmap \
        model.data.data_prefix=[1.0,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src] \
        model.data.splits_string=\'\"800,100,100\"\' \
        model.data.whole_word_masking=False \
        model.tokenizer.library=sentencepiece \
        model.tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
        ++model.hiddens.enc_output_name=z \
        ++model.hiddens.transform.q_z_given_x.cls_name=cond_gaussian \
        ++model.hiddens.transform.q_z_given_x.hidden_size=64 \
        ++model.hiddens.loss.mim.cls_name=a_mim \
        ++model.hiddens.loss.mim.loss_weight=0.5"
        // Change val_check_interval to 1 for resume as the len(dataloder) is 1 due to max_steps being the same as that of training and Lightning 2.0 raises an error
        // if val_check_interval > len(dataloder: https://github.com/Lightning-AI/lightning/blob/2.0.6/src/lightning/pytorch/loops/fit_loop.py#L259 at the beginning of fit_loop.run()
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/megatron_mim_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.arch=perceiver \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.micro_batch_size=2 \
        model.global_batch_size=4 \
        model.data.data_impl=text_mmap \
        model.data.data_prefix=[1.0,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src] \
        model.data.splits_string=\'\"800,100,100\"\' \
        model.data.whole_word_masking=False \
        model.tokenizer.library=sentencepiece \
        model.tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
        ++model.hiddens.enc_output_name=z \
        ++model.hiddens.transform.q_z_given_x.cls_name=cond_gaussian \
        ++model.hiddens.transform.q_z_given_x.hidden_size=64 \
        ++model.hiddens.loss.mim.cls_name=a_mim \
        ++model.hiddens.loss.mim.loss_weight=0.5"
        sh "rm -rf examples/nlp/language_modeling/megatron_mim_results"
      }
    }
    // stage('L2: NMT Bottleneck Fallback') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('L2: seq2seq (no bottleneck)') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=nll \
    //           model.encoder.arch=seq2seq \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref \
    //           model.validation_ds.src_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.src] \
    //           model.validation_ds.tgt_file_name=[/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref,/home/TestData/nlp/nmt/toy_data/wmt14-en-de.ref] \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt13-en-de.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt13-en-de.ref \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           trainer.devices=[1] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //   }
    // }
    // stage('L2: NMT Bottleneck Architecture') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('Bridge Encoder (identity)') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=nll \
    //           model.encoder.arch=bridge \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=identity \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //		 trainer.devices=[0] \
    // 		 trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //     stage('Perceiver Encoder (params)') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=nll \
    //           model.encoder.arch=perceiver \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           trainer.devices=[1] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //   }
    // }
    // stage('L2: NMT Bottleneck LVM') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   parallel {
    //     stage('VAE') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=vae \
    //           model.encoder.arch=perceiver \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           trainer.devices=[0] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //     stage('MIM') {
    //         steps {
    //           sh 'cd examples/nlp/machine_translation && \
    //           enc_dec_nmt-bottleneck.py \
    //           --config-path=conf \
    //           --config-name=aayn_bottleneck \
    //           do_testing=true \
    //           model.model_type=mim \
    //           model.encoder.arch=perceiver \
    //           model.encoder.hidden_steps=1 \
    //           model.encoder.hidden_blocks=1 \
    //           model.encoder.hidden_init_method=params \
    //           model.encoder.hidden_size=64 \
    //           model.encoder.inner_size=128 \
    //           model.encoder.num_attention_heads=2 \
    //           model.encoder.num_layers=2 \
    //           model.decoder.hidden_size=64 \
    //           model.decoder.inner_size=128 \
    //           model.decoder.num_attention_heads=2 \
    //           model.decoder.num_layers=2 \
    //           model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
    //           model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
    //           model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/spm_4k_ende.model \
    //           trainer.devices=[1] \
    //           trainer.accelerator="gpu" \
    //           +trainer.fast_dev_run=true \
    //           +trainer.limit_test_batches=2 \
    //           exp_manager=null \
    //           '
    //         }
    //     }
    //   }
    // }
    stage('L2: Megatron Bert Pretraining and Resume Training with Pipeline Paralleism') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/bert_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/bert_index_mappings"
      }
    }
    stage('L2: Megatron Bert Pretraining and Resume Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.sequence_parallel=True \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/bert_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/bert_index_mappings"
      }
    }
    stage('L2: Megatron Core Bert Pretraining and Resume Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "NVTE_FLASH_ATTN=0 python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=32 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        model.mcore_bert=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.sequence_parallel=True \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "NVTE_FLASH_ATTN=0 python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=20 \
        trainer.precision=32 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.mcore_bert=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/bert_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/bert_index_mappings"
      }
    }
    stage('L2: NeMo Bert Embedding Finetuning and Resume') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "NVTE_FLASH_ATTN=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python examples/nlp/information_retrieval/megatron_bert_embedding_finetuning.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.max_steps=12 \
        trainer.val_check_interval=4 \
        trainer.max_epochs=1 \
        +trainer.num_sanity_val_steps=0 \
        restore_from_path=/home/TestData/nlp/bert_embedding/bert_embedding_nemo_tiny.nemo \
        model.num_layers=2 \
        model.hidden_size=64 \
        model.ffn_hidden_size=256 \
        model.num_attention_heads=2 \
        model.megatron_legacy=False \
        model.mcore_bert=False \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.optim.lr=0.0005 \
        model.encoder_seq_length=512 \
        model.tokenizer.library='huggingface' \
        model.tokenizer.type='intfloat/e5-large-unsupervised' \
        model.data.data_train=/home/TestData/nlp/bert_embedding/bert_embedding_toy_data.jsonl \
        model.data.hard_negatives_to_train=4 \
        exp_manager.explicit_log_dir=examples/nlp/information_retrieval/bert_embedding_results \
        exp_manager.create_wandb_logger=False \
        exp_manager.resume_if_exists=False"
        sh "NVTE_FLASH_ATTN=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python examples/nlp/information_retrieval/megatron_bert_embedding_finetuning.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.max_steps=36 \
        trainer.val_check_interval=4 \
        trainer.max_epochs=1 \
        +trainer.num_sanity_val_steps=0 \
        restore_from_path=/home/TestData/nlp/bert_embedding/bert_embedding_nemo_tiny.nemo \
        model.num_layers=2 \
        model.hidden_size=64 \
        model.ffn_hidden_size=256 \
        model.num_attention_heads=2 \
        model.megatron_legacy=False \
        model.mcore_bert=False \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.optim.lr=0.0005 \
        model.encoder_seq_length=512 \
        model.tokenizer.library='huggingface' \
        model.tokenizer.type='intfloat/e5-large-unsupervised' \
        model.data.data_train=/home/TestData/nlp/bert_embedding/bert_embedding_toy_data.jsonl \
        model.data.hard_negatives_to_train=4 \
        exp_manager.explicit_log_dir=examples/nlp/information_retrieval/bert_embedding_results \
        exp_manager.create_wandb_logger=False \
        exp_manager.resume_if_exists=True"
        sh "rm -rf examples/nlp/information_retrieval/bert_embedding_results"
      }
    }
    stage('L2: Megatron Core Bert Embedding Finetuning and Resume') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "NVTE_FLASH_ATTN=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python examples/nlp/information_retrieval/megatron_bert_embedding_finetuning.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.max_steps=12 \
        trainer.val_check_interval=4 \
        trainer.max_epochs=36 \
        +trainer.num_sanity_val_steps=0 \
        restore_from_path=/home/TestData/nlp/bert_embedding/bert_embedding_mcore_tiny.nemo \
        model.num_layers=2 \
        model.hidden_size=64 \
        model.ffn_hidden_size=256 \
        model.num_attention_heads=2 \
        model.megatron_legacy=False \
        model.mcore_bert=True \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.optim.lr=0.0005 \
        model.encoder_seq_length=512 \
        model.tokenizer.library='huggingface' \
        model.tokenizer.type='intfloat/e5-large-unsupervised' \
        model.data.data_train=/home/TestData/nlp/bert_embedding/bert_embedding_toy_data.jsonl \
        model.data.hard_negatives_to_train=4 \
        exp_manager.explicit_log_dir=examples/nlp/information_retrieval/bert_embedding_results \
        exp_manager.create_wandb_logger=False \
        exp_manager.resume_if_exists=False"
        sh "NVTE_FLASH_ATTN=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FUSED_ATTN=0 python examples/nlp/information_retrieval/megatron_bert_embedding_finetuning.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.max_steps=16 \
        trainer.val_check_interval=4 \
        trainer.max_epochs=1 \
        +trainer.num_sanity_val_steps=0 \
        restore_from_path=/home/TestData/nlp/bert_embedding/bert_embedding_mcore_tiny.nemo \
        model.num_layers=2 \
        model.hidden_size=64 \
        model.ffn_hidden_size=256 \
        model.num_attention_heads=2 \
        model.megatron_legacy=False \
        model.mcore_bert=True \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.optim.lr=0.0005 \
        model.encoder_seq_length=512 \
        model.tokenizer.library='huggingface' \
        model.tokenizer.type='intfloat/e5-large-unsupervised' \
        model.data.data_train=/home/TestData/nlp/bert_embedding/bert_embedding_toy_data.jsonl \
        model.data.hard_negatives_to_train=4 \
        exp_manager.explicit_log_dir=examples/nlp/information_retrieval/bert_embedding_results \
        exp_manager.create_wandb_logger=False \
        exp_manager.resume_if_exists=True"
        sh "rm -rf examples/nlp/information_retrieval/bert_embedding_results"
      }
    }
    stage('L2: Megatron RETRO Pretraining and Resume Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_retro_pretraining.py \
        trainer.devices=2 \
        trainer.num_nodes=1 \
        trainer.accelerator=gpu \
        trainer.accumulate_grad_batches=1 \
        trainer.limit_val_batches=2 \
        exp_manager.resume_if_exists=True \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        trainer.val_check_interval=10 \
        exp_manager.exp_dir=examples/nlp/language_modeling/retro_results \
        model.data.data_prefix='' \
        model.data.knn_index='' \
        model.data.retrieval_prefix='' \
        model.tensor_model_parallel_size=2 \
        model.micro_batch_size=4 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.chunk_size=32 \
        model.enc_num_layers=2 \
        model.dec_num_layers=2 \
        model.enc_cross_attention=[1] \
        model.dec_cross_attention=[1] \
        +model.data.mock=True"
        sh "python examples/nlp/language_modeling/megatron_retro_pretraining.py \
        trainer.devices=2 \
        trainer.num_nodes=1 \
        trainer.accelerator=gpu \
        trainer.accumulate_grad_batches=1 \
        trainer.limit_val_batches=2 \
        exp_manager.resume_if_exists=True \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        trainer.val_check_interval=10 \
        exp_manager.exp_dir=examples/nlp/language_modeling/retro_results \
        model.data.data_prefix='' \
        model.data.knn_index='' \
        model.data.retrieval_prefix='' \
        model.tensor_model_parallel_size=2 \
        model.micro_batch_size=4 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.chunk_size=32 \
        model.enc_num_layers=2 \
        model.dec_num_layers=2 \
        model.enc_cross_attention=[1] \
        model.dec_cross_attention=[1] \
        +model.data.mock=True"
        sh "rm -rf examples/nlp/language_modeling/retro_results"
      }
    }
    stage('L2: Megatron RETRO muTransfer Pretraining Performance') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
            sh "python examples/nlp/language_modeling/megatron_retro_mutransfer_pretrain.py \
                trainer.devices=2 \
                trainer.num_nodes=1 \
                trainer.accelerator=gpu \
                trainer.accumulate_grad_batches=1 \
                trainer.max_steps=100 \
                trainer.log_every_n_steps=1 \
                trainer.precision=16 \
                trainer.val_check_interval=100 \
                trainer.limit_val_batches=0 \
                trainer.gradient_clip_val=1.0 \
                +trainer.num_sanity_val_steps=0 \
                exp_manager.exp_dir=examples/nlp/language_modeling/retro_results/ \
                +exp_manager.version=smalltest \
                model.data.neighbors=2 \
                model.megatron_amp_O2=False \
                model.apply_query_key_layer_scaling=False \
                model.tensor_model_parallel_size=1 \
                model.optim.name=muadamw \
                model.optim.weight_decay=0.1 \
                model.optim.betas=[0.9,0.95] \
                model.optim.lr=6e-4 \
                model.optim.sched.warmup_steps=1000 \
                model.optim.sched.constant_steps=0 \
                model.optim.sched.min_lr=6e-5 \
                model.add_position_embedding=False \
                model.enc_num_layers=2 \
                model.dec_num_layers=6 \
                model.enc_cross_attention=[0] \
                model.dec_cross_attention=[3,5] \
                model.hidden_size=96 \
                model.ffn_hidden_size=384 \
                model.init_method_std=0.023 \
                model.num_attention_heads=12 \
                model.max_position_embeddings=1024 \
                model.encoder_seq_length=1024 \
                model.tokenizer.library=megatron \
                model.tokenizer.type=GPT2BPETokenizer \
                model.tokenizer.merge_file=/home/TestData/nlp/megatron_retro/gpt2-merges.txt \
                model.tokenizer.vocab_file=/home/TestData/nlp/megatron_retro/gpt2-vocab.json \
                model.data.data_prefix=[/home/TestData/nlp/megatron_retro/retro_wiki_test_text_document] \
                model.data.knn_index=[/home/TestData/nlp/megatron_retro/knn2_map_wiki_test.idx] \
                model.data.retrieval_prefix=/home/TestData/nlp/megatron_retro/retro_wiki_test_text_document \
                model.data.index_mapping_dir=/home/TestData/nlp/megatron_retro \
                model.data.num_workers=8 \
                model.micro_batch_size=8 \
                model.normalization=rmsnorm \
                model.transformer_block_type=pre_ln \
                model.bias_activation_fusion=True \
                model.bias_dropout_add_fusion=False \
                model.masked_softmax_fusion=True \
                model.hidden_dropout=0 \
                model.attention_dropout=0 \
                model.fp32_residual_connection=True \
                model.shape_file=/home/TestData/nlp/megatron_retro/o1_rel_shape_info_tiny.yaml"
        sh '''python -c "import pandas as pd
import pathlib
from pandas.testing import assert_frame_equal
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
if not (torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()):
    import sys
    sys.exit(0)
event_file = list(pathlib.Path('examples/nlp/language_modeling/retro_results/megatron_retro/smalltest').glob('events.out.tfevents*'))[0]
ea = EventAccumulator(str(event_file)).Reload()
vals = []
for i in ea.Scalars('reduced_train_loss'):
    vals.append(i.value)
training_curve = pd.DataFrame({'loss': vals})
gt_curve = pd.read_csv('/home/TestData/nlp/megatron_retro/expected_learning_curve.csv')
assert_frame_equal(training_curve, gt_curve, rtol=1e-3, atol=1e-3)"'''
        sh "rm -rf examples/nlp/language_modeling/retro_results"
      }
    }
    stage('L2: BioMegatron Bert NER Task') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/token_classification/token_classification_train.py \
        exp_manager.exp_dir=examples/nlp/language_modeling/token_classification_results \
        trainer.max_epochs=1 \
        model.dataset.data_dir=/home/TestData/nlp/ner \
        model.language_model.pretrained_model_name=biomegatron345m_biovocab_30k_cased \
        model.tokenizer.tokenizer_name=null"
        sh "rm -rf examples/nlp/language_modeling/token_classification_results"
      }
    }
    stage('L2: Megatron GPT Pretraining and Resume Training TETransformerLayerTP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        ++model.name=megatron_gpt_full_te_layer_autocast \
        model.mcore_gpt=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=layernorm1p \
        model.bias_activation_fusion=True \
        model.bias_dropout_add_fusion=True \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method=null \
        model.activations_checkpoint_granularity=null \
        model.activations_checkpoint_num_layers=null \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        exp_manager.resume_if_exists=True \
        ++model.name=megatron_gpt_full_te_layer_autocast \
        model.mcore_gpt=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=layernorm1p \
        model.bias_activation_fusion=True \
        model.bias_dropout_add_fusion=True \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method=null \
        model.activations_checkpoint_granularity=null \
        model.activations_checkpoint_num_layers=null \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    // @chcui: model.cpu_offloading_num_layers=7 # temp workaround before m-lm !1124 is merged
    // @athitten: Revert limit_val_batches to 2 until limit_val_batches 1.0 leading to no validation is fixed for non DictConfig data_prefix
    stage('L2: Megatron GPT Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }

    stage('L2: Megatron GPT Pretraining and Resume Training TP=2 with Torch Distributed Checkpoint') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.mcore_gpt=True \
        model.torch_distributed_checkpoint=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=distributed_fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.mcore_gpt=True \
        model.torch_distributed_checkpoint=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=distributed_fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
/*
    stage('L2: Megatron GPT Pretraining with EP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.expert_model_parallel_size=2 \
        ++model.num_moe_experts=2 \
        ++model.moe_router_topk=1 \
        ++model.megatron_amp_O2=True \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
*/
    stage('L2: Megatron GPT with Rope Pretraining and Resume Training TP=2') {
     when {
       anyOf {
         branch 'main'
         changeRequest target: 'main'
       }
     }
     failFast true
     steps {
       sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
       trainer.devices=2 \
       trainer.accelerator=gpu \
       trainer.log_every_n_steps=1 \
       trainer.val_check_interval=2 \
       trainer.limit_val_batches=2 \
       trainer.accumulate_grad_batches=1 \
       trainer.max_steps=3 \
       trainer.precision=16 \
       trainer.gradient_clip_val=1.0 \
       exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
       model.tensor_model_parallel_size=2 \
       model.optim.name=fused_adam \
       model.optim.lr=2e-4 \
       model.optim.sched.warmup_steps=1 \
       model.optim.sched.constant_steps=1 \
       model.optim.sched.min_lr=8e-5 \
       model.max_position_embeddings=128 \
       model.encoder_seq_length=128 \
       model.data.seq_length=128 \
       model.position_embedding_type=rope \
       model.rotary_percentage=0.5 \
       model.normalization=rmsnorm \
       model.bias=False \
       model.bias_activation_fusion=False \
       model.bias_dropout_add_fusion=False \
       model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
       model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
       model.num_layers=8 \
       model.cpu_offloading_num_layers=7 \
       model.hidden_size=256 \
       model.num_attention_heads=8 \
       model.activations_checkpoint_method='block' \
       model.activations_checkpoint_granularity='full' \
       model.activations_checkpoint_num_layers=1 \
       model.data.data_prefix='{train:[1.0,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document],validation:[/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document],test:[/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document]}' \
       model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        // commented out to save time on github ci @adithyare
        //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        //trainer.devices=2 \
        //trainer.accelerator=gpu \
        //trainer.log_every_n_steps=1 \
        //trainer.val_check_interval=2 \
        //trainer.limit_val_batches=1 \
        //trainer.accumulate_grad_batches=1 \
        //trainer.max_steps=6 \
        //trainer.precision=16 \
        //trainer.gradient_clip_val=1.0 \
        //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        //exp_manager.resume_if_exists=True \
        //model.tensor_model_parallel_size=2 \
        //model.optim.name=fused_adam \
        //model.optim.lr=2e-4 \
        //model.optim.sched.warmup_steps=2 \
        //model.optim.sched.constant_steps=2 \
        //model.optim.sched.min_lr=8e-5 \
        //model.max_position_embeddings=128 \
        //model.encoder_seq_length=128 \
        //model.data.seq_length=128 \
        //model.position_embedding_type=rope \
        //model.rotary_percentage=0.5 \
        //model.normalization=rmsnorm \
        //model.bias=False \
        //model.bias_activation_fusion=False \
        //model.bias_dropout_add_fusion=False \
        //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        //model.num_layers=8 \
        //model.hidden_size=256 \
        //model.num_attention_heads=8 \
        //model.activations_checkpoint_method='block' \
        //model.activations_checkpoint_granularity='full' \
        //model.activations_checkpoint_num_layers=1 \
        //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
       sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
       sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
     }

    // This test requires Ampere but some of the test GPUs are Volta
    // Need to add a check for compute capability before uncommenting this test
    // stage('L2: Megatron GPT with Rope Pretraining using Flash Attention and Resume Training TP=2') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    //     trainer.devices=2 \
    //     trainer.accelerator=gpu \
    //     trainer.log_every_n_steps=1 \
    //     trainer.val_check_interval=2 \
    //     trainer.limit_val_batches=2 \
    //     trainer.accumulate_grad_batches=1 \
    //     trainer.max_steps=3 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
    //     model.tensor_model_parallel_size=2 \
    //     model.optim.name=fused_adam \
    //     model.optim.lr=2e-4 \
    //     model.optim.sched.warmup_steps=1 \
    //     model.optim.sched.constant_steps=1 \
    //     model.optim.sched.min_lr=8e-5 \
    //     model.max_position_embeddings=128 \
    //     model.encoder_seq_length=128 \
    //     model.data.seq_length=128 \
    //     model.position_embedding_type=rope \
    //     model.rotary_percentage=0.5 \
    //     model.normalization=rmsnorm \
    //     model.bias=False \
    //     model.bias_activation_fusion=False \
    //     model.bias_dropout_add_fusion=False \
    //     model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    //     model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    //     model.num_layers=8 \
    //     model.hidden_size=256 \
    //     model.num_attention_heads=8 \
    //     model.activations_checkpoint_method='block' \
    //     model.activations_checkpoint_granularity='full' \
    //     model.activations_checkpoint_num_layers=1 \
    //     model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
    //     model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings \
    //     model.use_flash_attention=True "
    //     // commented out to save time on github ci @adithyare
    //     //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    //     //trainer.devices=2 \
    //     //trainer.accelerator=gpu \
    //     //trainer.log_every_n_steps=1 \
    //     //trainer.val_check_interval=2 \
    //     //trainer.limit_val_batches=1 \
    //     //trainer.accumulate_grad_batches=1 \
    //     //trainer.max_steps=6 \
    //     //trainer.precision=16 \
    //     //trainer.gradient_clip_val=1.0 \
    //     //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
    //     //exp_manager.resume_if_exists=True \
    //     //model.tensor_model_parallel_size=2 \
    //     //model.optim.name=fused_adam \
    //     //model.optim.lr=2e-4 \
    //     //model.optim.sched.warmup_steps=2 \
    //     //model.optim.sched.constant_steps=2 \
    //     //model.optim.sched.min_lr=8e-5 \
    //     //model.max_position_embeddings=128 \
    //     //model.encoder_seq_length=128 \
    //     //model.data.seq_length=128 \
    //     //model.position_embedding_type=rope \
    //     //model.rotary_percentage=0.5 \
    //     //model.normalization=rmsnorm \
    //     //model.bias=False \
    //     //model.bias_activation_fusion=False \
    //     //model.bias_dropout_add_fusion=False \
    //     //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    //     //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    //     //model.num_layers=8 \
    //     //model.hidden_size=256 \
    //     //model.num_attention_heads=8 \
    //     //model.activations_checkpoint_method='block' \
    //     //model.activations_checkpoint_granularity='full' \
    //     //model.activations_checkpoint_num_layers=1 \
    //     //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
    //     //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings \
    //     //model.use_flash_attention=True"
    //     sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
    //     sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
    //   }
    // }
    // @chcui: model.cpu_offloading_num_layers=7 # temp workaround before m-lm !1124 is merged
    stage('L2: Megatron GPT with ALiBi Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.position_embedding_type=alibi \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        // not testing resume functionality to save time on ci @adithyare
        //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        //trainer.devices=2 \
        //trainer.accelerator=gpu \
        //trainer.log_every_n_steps=1 \
        //trainer.val_check_interval=2 \
        //trainer.limit_val_batches=1 \
        //trainer.accumulate_grad_batches=1 \
        //trainer.max_steps=6 \
        //trainer.precision=16 \
        //trainer.gradient_clip_val=1.0 \
        //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        //exp_manager.resume_if_exists=True \
        //model.tensor_model_parallel_size=2 \
        //model.optim.name=fused_adam \
        //model.optim.lr=2e-4 \
        //model.optim.sched.warmup_steps=2 \
        //model.optim.sched.constant_steps=2 \
        //model.optim.sched.min_lr=8e-5 \
        //model.max_position_embeddings=128 \
        //model.encoder_seq_length=128 \
        //model.data.seq_length=128 \
        //model.position_embedding_type=alibi \
        //model.normalization=rmsnorm \
        //model.bias=False \
        //model.bias_activation_fusion=False \
        //model.bias_dropout_add_fusion=False \
        //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        //model.num_layers=8 \
        //model.hidden_size=256 \
        //model.num_attention_heads=8 \
        //model.activations_checkpoint_method='block' \
        //model.activations_checkpoint_granularity='full' \
        //model.activations_checkpoint_num_layers=1 \
        //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    // @chcui: model.cpu_offloading_num_layers=7 # temp workaround before m-lm !1124 is merged
    stage('L2: Megatron GPT with KERPLE Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.position_embedding_type=kerple \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        // commented out to save time on github ci @adithyare
        //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        //trainer.devices=2 \
        //trainer.accelerator=gpu \
        //trainer.log_every_n_steps=1 \
        //trainer.val_check_interval=2 \
        //trainer.limit_val_batches=1 \
        //trainer.accumulate_grad_batches=1 \
        //trainer.max_steps=6 \
        //trainer.precision=16 \
        //trainer.gradient_clip_val=1.0 \
        //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        //exp_manager.resume_if_exists=True \
        //model.tensor_model_parallel_size=2 \
        //model.optim.name=fused_adam \
        //model.optim.lr=2e-4 \
        //model.optim.sched.warmup_steps=2 \
        //model.optim.sched.constant_steps=2 \
        //model.optim.sched.min_lr=8e-5 \
        //model.max_position_embeddings=128 \
        //model.encoder_seq_length=128 \
        //model.data.seq_length=128 \
        //model.position_embedding_type=kerple \
        //model.normalization=rmsnorm \
        //model.bias=False \
        //model.bias_activation_fusion=False \
        //model.bias_dropout_add_fusion=False \
        //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        //model.num_layers=8 \
        //model.hidden_size=256 \
        //model.num_attention_heads=8 \
        //model.activations_checkpoint_method='block' \
        //model.activations_checkpoint_granularity='full' \
        //model.activations_checkpoint_num_layers=1 \
        //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    // @chcui: model.cpu_offloading_num_layers=7 # temp workaround before m-lm !1124 is merged
    stage('L2: Megatron GPT Pretraining and Resume Training PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=bf16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.mcore_gpt=True \
        model.megatron_amp_O2=True \
        model.optim.name=distributed_fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.activation=fast-swiglu \
        model.bias_activation_fusion=False \
        model.hidden_dropout=0.0 \
        model.attention_dropout=0.0 \
        model.transformer_block_type=normformer \
        model.headscale=True \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=bf16 \
        trainer.gradient_clip_val=1.0 \
        model.mcore_gpt=True \
        model.megatron_amp_O2=True \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.optim.name=distributed_fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.activation=fast-swiglu \
        model.bias_activation_fusion=False \
        model.hidden_dropout=0.0 \
        model.attention_dropout=0.0 \
        model.transformer_block_type=normformer \
        model.headscale=True \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.cpu_offloading_num_layers=7 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    stage('L2: Megatron GPT Finetuning PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        +trainer.limit_val_batches=2 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_sft_results \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.peft.peft_scheme=null \
        model.data.train_ds.micro_batch_size=1 \
        model.data.train_ds.global_batch_size=4 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[0.3,0.7] \
        model.data.train_ds.num_workers=0 \
        model.data.test_ds.micro_batch_size=1 \
        model.data.test_ds.global_batch_size=1 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.test_ds.names=[quarel] \
        model.data.validation_ds.micro_batch_size=1 \
        model.data.validation_ds.global_batch_size=1 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.validation_ds.names=[quarel,trec]"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        +trainer.limit_val_batches=2 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_sft_results \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.peft.peft_scheme=null \
        model.data.train_ds.micro_batch_size=1 \
        model.data.train_ds.global_batch_size=4 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[0.3,0.7] \
        model.data.train_ds.num_workers=0 \
        model.data.test_ds.micro_batch_size=1 \
        model.data.test_ds.global_batch_size=1 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.test_ds.names=[quarel] \
        model.data.validation_ds.micro_batch_size=1 \
        model.data.validation_ds.global_batch_size=1 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.validation_ds.names=[quarel,trec]"
        sh "rm -rf examples/nlp/language_modeling/gpt_sft_results"
      }
    }
    stage('L2: Megatron GPT Finetuning StarCoder PP=1') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
        trainer.devices=1 \
        trainer.num_nodes=1 \
        trainer.precision=32 \
        trainer.max_steps=4 \
        trainer.val_check_interval=4 \
        trainer.enable_checkpointing=False \
        +trainer.limit_val_batches=2 \
        +trainer.limit_test_batches=2 \
        exp_manager.checkpoint_callback_params.save_best_model=False \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_sft_results \
        model.optim.name=distributed_fused_adam \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/starcoder-ci-nemo/megatron_starcoder_tp1_pp1.nemo \
        model.tensor_model_parallel_size=1 \
        model.pipeline_model_parallel_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.num_workers=0 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.test_ds.num_workers=0 \
        model.data.train_ds.concat_sampling_probabilities=[1.0]"
        sh "rm -rf examples/nlp/language_modeling/gpt_sft_results"
      }
    }
    stage('L2: Megatron GPT PEFT Lora PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf examples/nlp/language_modeling/gpt_peft_lora_results_pp2"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.max_epochs=9999 \
        trainer.max_steps=3 \
        trainer.val_check_interval=3 \
        ++trainer.limit_val_batches=2 \
        trainer.precision=16 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_peft_lora_results_pp2 \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
        model.peft.peft_scheme='lora' \
        model.answer_only_loss=True \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "rm -rf examples/nlp/language_modeling/gpt_peft_lora_results_pp2"
      }
    }
    stage('L2: Megatron GPT PEFT Lora TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/nlp/lora_tuning_tp2"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.max_epochs=9999 \
        trainer.max_steps=3 \
        trainer.val_check_interval=3 \
        ++trainer.limit_val_batches=2 \
        trainer.precision=16 \
        exp_manager.exp_dir=/home/TestData/nlp/lora_tuning_tp2 \
        model.pipeline_model_parallel_size=1 \
        model.tensor_model_parallel_size=2 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
        model.peft.peft_scheme='lora' \
        model.answer_only_loss=True \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
        model.peft.restore_from_path=/home/TestData/nlp/lora_tuning_tp2/megatron_gpt_peft_lora_tuning/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
        model.tensor_model_parallel_size=2 \
        trainer.devices=2 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel_4.jsonl] \
        model.data.test_ds.names=['quarel4'] \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.data.test_ds.tokens_to_generate=10 \
        model.data.test_ds.write_predictions_to_file=True \
        model.data.test_ds.output_file_path_prefix='/home/TestData/nlp/lora_tuning_tp2/out' \
        inference.greedy=True \
        inference.repetition_penalty=1.0 \
        inference.outfile_path='/home/TestData/nlp/lora_tuning_tp2/out.jsonl'"
        sh "rm -rf /home/TestData/nlp/lora_tuning_tp2"
      }
    }
    stage('L2: Megatron GPT PEFT Lora TP=2 SP') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/nlp/lora_tuning_tp2_sp"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.max_epochs=9999 \
        trainer.max_steps=3 \
        trainer.val_check_interval=3 \
        ++trainer.limit_val_batches=2 \
        trainer.precision=16 \
        exp_manager.exp_dir=/home/TestData/nlp/lora_tuning_tp2 \
        model.pipeline_model_parallel_size=1 \
        model.tensor_model_parallel_size=2 \
        model.sequence_parallel=true \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
        model.peft.peft_scheme='lora' \
        model.answer_only_loss=True \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "rm -rf /home/TestData/nlp/lora_tuning_tp2_sp"
      }
    }
    stage('L2: Megatron GPT Eval') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
            gpt_model_file=/home/TestData/nlp/megatron_gpt/125M/megatron_gpt.nemo \
            prompts=['How to fix GPU memory? A:'] \
            tensor_model_parallel_size=1 \
            inference.tokens_to_generate=32 \
            trainer.precision=32"
      }
    }
    stage('L2: Megatron GPT Eval PP2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
            gpt_model_file=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
            server=False \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=2 \
            trainer.devices=2 \
            trainer.num_nodes=1 \
            trainer.precision=32"
      }
    }
    stage('L2: Megatron GPT SFT Eval (inference seq len > training seq len)') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
            model.restore_from_path=/home/TestData/nlp/megatron_gpt_sft/megatron_gpt_rope_sft.nemo \
            model.peft.restore_from_path=null \
            model.data.test_ds.file_names=['/home/TestData/nlp/megatron_gpt_sft/sample.jsonl'] \
            model.data.test_ds.names=['test'] \
            model.data.test_ds.global_batch_size=1 \
            model.data.test_ds.micro_batch_size=1 \
            model.data.test_ds.tokens_to_generate=30 \
            model.data.test_ds.max_seq_length=6000 \
            model.data.test_ds.write_predictions_to_file=True \
            model.data.test_ds.output_file_path_prefix='examples/nlp/language_modeling/out' \
            inference.greedy=True \
            inference.repetition_penalty=1.0 \
            inference.outfile_path='examples/nlp/language_modeling/out.jsonl' && \
            rm -rf examples/nlp/language_modeling/out.jsonl"
      }
    }

    // TODO: Add this test back. Test was failing on CI machines due to HW error
    // stage('L2: Megatron GPT Convert from Megatron-LM checkpoing and Eval') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh "python -m torch.distributed.launch --nproc_per_node=2 \
    //     examples/nlp/language_modeling/megatron_lm_ckpt_to_nemo.py \
    //     --checkpoint_folder=/home/TestData/nlp/megatron_gpt/data/gpt/iter_0008700 \
    //     --checkpoint_name=model_optim_rng.pt \
    //     --hparams_file=/home/TestData/nlp/megatron_gpt/data/gpt/iter_0008700/hparams.yaml \
    //     --nemo_file_path=examples/nlp/language_modeling/small_gpt.nemo \
    //     --model_type=gpt \
    //     --pipeline_model_parallel_size=1 \
    //     --gpus_per_node=2 \
    //     --tensor_model_parallel_size=2"
    //     sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
    //     --gpt_model_file=examples/nlp/language_modeling/small_gpt.nemo \
    //     --tokens_to_generate=32 \
    //     --tensor_model_parallel_size=2 \
    //     --prompt='This is a test.'"
    //     sh "rm examples/nlp/language_modeling/small_gpt.nemo"
    //   }
    // }
    stage('L2: Megatron Change Partitions') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel{
        stage('Reduce TP Num Partitions (2 to 1) and PP Num Partitions (1 to 2)'){
          steps{
            sh "python examples/nlp/language_modeling/megatron_change_num_partitions.py \
                --model_file \
                /home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
                --target_file \
                /home/TestData/nlp/megatron_gpt/TP2-Temp/test-reduce.nemo \
                --tensor_model_parallel_size \
                2 \
                --target_tensor_model_parallel_size \
                1 \
                --pipeline_model_parallel_size \
                1 \
                --target_pipeline_model_parallel_size \
                2"
            sh "rm /home/TestData/nlp/megatron_gpt/TP2-Temp/test-reduce.nemo"
          }
        }
        stage('Increase TP Num Partitions (2 to 4) and PP Num Partitions (1 to 2)'){
          steps{
            sh "python examples/nlp/language_modeling/megatron_change_num_partitions.py \
                --model_file \
                /home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
                --target_file \
                /home/TestData/nlp/megatron_gpt/TP2-Temp/test-increase.nemo \
                --tensor_model_parallel_size \
                2 \
                --target_tensor_model_parallel_size \
                4 \
                --pipeline_model_parallel_size \
                1 \
                --target_pipeline_model_parallel_size \
                1"
            sh "rm /home/TestData/nlp/megatron_gpt/TP2-Temp/test-increase.nemo"
          }
        }
      }
    }
    stage('L2: Megatron T5 Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=relative \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='fast-swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=relative \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='fast-swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 with ALiBi Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=alibi \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=alibi \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 with KERPLE Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=kerple \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=kerple \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 Pretraining and Resume Training PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.decoder.num_layers=1 \
        model.encoder.hidden_size=64 \
        model.decoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.decoder.num_attention_heads=8 \
        model.decoder.ffn_hidden_size=2048 \
        model.encoder.activation='gelu' \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='post_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.decoder.num_layers=1 \
        model.encoder.hidden_size=64 \
        model.decoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.decoder.num_attention_heads=8 \
        model.decoder.ffn_hidden_size=2048 \
        model.encoder.activation='gelu' \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='post_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 w/ Mixture of Expert Pretraining') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.decoder.num_layers=1 \
        model.encoder.num_moe_experts=4 \
        model.decoder.num_moe_experts=4 \
        model.encoder.moe_frequency=3 \
        model.decoder.moe_frequency=1 \
        model.encoder.hidden_size=64 \
        model.decoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.decoder.num_attention_heads=8 \
        model.decoder.ffn_hidden_size=2048 \
        model.encoder.activation='gelu' \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='post_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }

    stage('L2: Megatron UL2 Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py -cn megatron_ul2_config \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='normformer' \
        model.encoder.headscale=True \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.decoder.transformer_block_type='normformer' \
        model.decoder.headscale=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='normformer' \
        model.encoder.headscale=True \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.decoder.transformer_block_type='normformer' \
        model.decoder.headscale=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 Eval') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/megatron_t5_eval.py \
            --model_file \
            /home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            --prompt \
            'How do I fix my GPU memory issue? I am seeing <mask> out of memory.' \
            --tensor_model_parallel_size 1"
      }
    }
    stage('L2: Megatron BART Pretraining and Resume Training, TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='reglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='reglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.data_prefix='{train:[1.0,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document],test:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document], validation:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]}'"
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=5 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='reglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='reglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.data_prefix='{train:[1.0,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document],test:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document], validation:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]}'"
        sh "rm -rf examples/nlp/language_modeling/bart_pretrain_results"
      }
    }
    stage('L2: Megatron BART Pretraining and Resume Training, PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='geglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.respect_document_boundaries=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='geglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.respect_document_boundaries=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "rm -rf examples/nlp/language_modeling/bart_pretrain_results"
      }
    }
    stage('L2: Megatron T5 GLUE/XNLI Finetuning') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        // TODO(Oktai15): update it in 1.8.0 version
        stage('T5 GLUE RTE') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=1 \
            +trainer.limit_val_batches=2 \
            +trainer.limit_test_batches=2 \
            trainer.accumulate_grad_batches=1 \
            trainer.max_steps=2 \
            trainer.precision=16 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_glue_results \
            model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            model.pipeline_model_parallel_size=1 \
            model.pipeline_model_parallel_split_rank=0 \
            model.data.train_ds.task_name=rte \
            model.data.train_ds.global_batch_size=4 \
            model.data.train_ds.micro_batch_size=2 \
            model.data.validation_ds.global_batch_size=2 \
            model.data.validation_ds.micro_batch_size=2 \
            model.data.train_ds.file_path=/home/TestData/nlp/megatron_t5/data/train_ci.tsv \
            model.data.validation_ds.task_name=rte \
            model.data.validation_ds.file_path=/home/TestData/nlp/megatron_t5/data/dev_ci.tsv \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_glue_results"
          }
        }
        stage('T5 GLUE XNLI') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py \
            -cn megatron_t5_config_finetune_glue_xnli \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=1 \
            +trainer.limit_val_batches=2 \
            +trainer.limit_test_batches=2 \
            trainer.accumulate_grad_batches=1 \
            trainer.max_steps=2 \
            trainer.precision=16 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_xnli_results \
            model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            model.pipeline_model_parallel_size=1 \
            model.pipeline_model_parallel_split_rank=0 \
            model.data.train_ds.global_batch_size=4 \
            model.data.train_ds.micro_batch_size=2 \
            model.data.validation_ds.global_batch_size=2 \
            model.data.validation_ds.micro_batch_size=2 \
            model.data.test_ds.global_batch_size=2 \
            model.data.test_ds.micro_batch_size=2 \
            model.data.train_ds.task_name=rte \
            model.data.train_ds.file_path=/home/TestData/nlp/megatron_t5/data/train_ci.tsv \
            model.data.validation_ds.task_name=xnli \
            model.data.validation_ds.file_path=/home/TestData/nlp/megatron_t5/data/xnli_dev_ci.tsv \
            model.data.test_ds.task_name=xnli \
            model.data.test_ds.file_path=/home/TestData/nlp/megatron_t5/data/xnli_dev_ci.tsv \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_xnli_results"
          }
        }
      }
    }

    stage('L2: Megatron T5 PEFT Lora TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/nlp/t5_lora_tuning_tp2"
        sh "python examples/nlp/language_modeling/tuning/megatron_t5_finetuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.max_epochs=9999 \
        trainer.max_steps=3 \
        trainer.val_check_interval=3 \
        ++trainer.limit_val_batches=2 \
        trainer.precision=16 \
        exp_manager.exp_dir=/home/TestData/nlp/t5_lora_tuning_tp2 \
        model.pipeline_model_parallel_size=1 \
        model.tensor_model_parallel_size=2 \
        model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m_tp2.nemo \
        model.peft.peft_scheme='lora' \
        model.answer_only_loss=True \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "python examples/nlp/language_modeling/tuning/megatron_t5_generate.py \
        model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m_tp2.nemo \
        model.peft.restore_from_path=/home/TestData/nlp/t5_lora_tuning_tp2/megatron_t5_peft_lora_tuning/checkpoints/megatron_t5_peft_lora_tuning.nemo \
        model.peft.restore_from_ckpt_name=null \
        model.peft.restore_from_hparams_path=null \
        model.tensor_model_parallel_size=2 \
        trainer.devices=2 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel_4.jsonl] \
        model.data.test_ds.names=['quarel4'] \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.data.test_ds.tokens_to_generate=10 \
        model.data.test_ds.write_predictions_to_file=True \
        model.data.test_ds.output_file_path_prefix='/home/TestData/nlp/t5_lora_tuning_tp2/out' \
        inference.greedy=True \
        inference.repetition_penalty=1.0 \
        inference.outfile_path='/home/TestData/nlp/t5_lora_tuning_tp2/out.jsonl'"
        sh "rm -rf /home/TestData/nlp/t5_lora_tuning_tp2"
      }
    }

    stage('L2: Megatron FIM Dataset') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=1 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        ++model.name=megatron_gpt_full_te_layer_autocast \
        model.mcore_gpt=True \
        model.tensor_model_parallel_size=1 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=layernorm1p \
        model.bias_activation_fusion=True \
        model.bias_dropout_add_fusion=True \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method=null \
        model.activations_checkpoint_granularity=null \
        model.activations_checkpoint_num_layers=null \
        model.data.data_prefix='[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document]' \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings \
        ++model.data.add_fim=True \
        ++model.data.fim.extra_tokens.prefix='fim_prefix' \
        ++model.data.fim.extra_tokens.middle='fim_middle' \
        ++model.data.fim.extra_tokens.suffix='fim_suffix' \
        ++model.data.fim.extra_tokens.pad='fim_pad' \
        ++model.data.fim.extra_tokens.eod='endoftext'"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
      }
    }
    
    stage('L2: Megatron Mock Data Generation') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('MockGPTDataset') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
            trainer.max_steps=10 \
            trainer.limit_val_batches=7 \
            trainer.val_check_interval=10 \
            exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
            model.data.data_impl=mock \
            model.data.data_prefix=[] \
            "
            sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
          }
        }
        stage('MockT5Dataset') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
            trainer.max_steps=10 \
            trainer.limit_val_batches=3 \
            trainer.val_check_interval=10 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
            model.data.data_impl=mock \
            model.data.data_prefix=[] \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
          }
        }
      }
    }

    stage('L2: TTS Fast dev runs 1') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      parallel {
        stage('Tacotron 2') {
          steps {
            sh 'python examples/tts/tacotron2.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.decoder.decoder_rnn_dim=256 \
            model.decoder.attention_rnn_dim=1024 \
            model.decoder.prenet_dim=128 \
            model.postnet.postnet_n_convolutions=3 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs \
            ~trainer.check_val_every_n_epoch \
            '
          }
        }
        stage('WaveGlow') {
          steps {
            sh 'python examples/tts/waveglow.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            model.waveglow.n_flows=4 \
            model.waveglow.n_wn_layers=2 \
            model.waveglow.n_wn_channels=32 \
            ~trainer.check_val_every_n_epoch'
          }
        }
        stage('FastPitch') {
          steps {
            sh 'python examples/tts/fastpitch.py \
            --config-name fastpitch_align_v1.05 \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/beta_priors \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.pitch_mean=212.35873413085938 \
            model.pitch_std=68.52806091308594 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            model.symbols_embedding_dim=64 \
            model.input_fft.d_inner=384 \
            model.input_fft.n_layer=2 \
            model.output_fft.d_inner=384 \
            model.output_fft.n_layer=2 \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('RADTTS') {
          steps {
            sh 'python examples/tts/radtts.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/radtts_beta_priors \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.pitch_mean=212.35873413085938 \
            model.pitch_std=68.52806091308594 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            export_dir=/home/TestData/radtts_test \
            model.optim.lr=0.0001 \
            model.modelConfig.decoder_use_partial_padding=True \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('Mixer-TTS') {
          steps {
            sh 'python examples/tts/mixer_tts.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/sup_data \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.pitch_mean=212.35873413085938 \
            model.pitch_std=68.52806091308594 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('Hifigan') {
          steps {
            sh 'python examples/tts/hifigan.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            +trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            model.generator.upsample_initial_channel=64 \
            +model.debug=true \
            ~trainer.check_val_every_n_epoch'
          }
        }
      }
    }
    stage('L2: NeRF') {
      when {
        anyOf {
          branch 'r1.21.0'
          changeRequest target: 'r1.21.0'
        }
      }
      parallel {
        stage('DreamFusion') {
          steps {
            sh 'python examples/multimodal/text_to_image/nerf/main.py \
            trainer.num_nodes=1 \
            trainer.devices="[0]" \
            trainer.max_steps=1000 \
            model.prompt="a DSLR photo of a delicious hamburger" \
            exp_manager.exp_dir=examples/multimodal/text_to_image/nerf/dreamfusion_results'
            sh 'rm -rf examples/multimodal/text_to_image/nerf/dreamfusion_results'
          }
        }
      }
    }
    stage('L??: Speech Checkpoints tests') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/speech_to_text_eval.py \
            pretrained_name=QuartzNet15x5Base-En  \
            dataset_manifest=/home/TestData/librispeech/librivox-dev-other.json \
            batch_size=64 \
            tolerance=0.1012'
        sh 'rm -f examples/asr/evaluation_transcripts.json'
      }
    }
  }

  post {
    always {
      sh 'chmod -R 777 .'
      cleanWs()
    }
  }
}
