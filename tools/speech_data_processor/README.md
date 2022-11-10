# Speech Data Processor

Speech Data Processor (SDP) is a toolkit to make it easy to:
1. write code to process a new dataset, minimizing the amount of boilerplate code required.
2. share the steps for processing a speech dataset. Sharing processing steps can be as easy as sharing a YAML file.

SDP's philosophy is to represent processing operations as 'processor' classes. Many common processing operations are provided, and it is easy to add your own. In some cases, all you will need to do to process a new dataset is simply to write a YAML file containing the parameters needed to process your dataset.

SDP is specifically intended for the use case when you have an existing dataset with the audio & text pairs already specified in some form, and you wish to create a JSON manifest suitable for use with NeMo. SDP allows for intermediate cleaning and filtering steps which involve amending the 'ground truth' `"text"` or dropping utterances which are deemed to be too inaccurate for training on.

## Quick intro to Speech Data Processor

* The steps to process a dataset are specified by a YAML config file.
* The YAML config file contains a list of processor classes & the args to pass into the constructor.
* Each processor class inputs an existing manifest (except for classes which create an 'initial' manifest from some external transcript file)  & outputs a modified version of the manifest. It may change other files in the process, e.g. resample audio.
* To process a manifest, you need to list the chain of processors you wish to use.
* If a processor is not included, you can make your own.

## YAML config file layout
A simplified version of an SDP file can be:

```yaml
processors: 

  # use existing classes for popular datasets or make your own class
  - _target_: sdp.processors.CreateInitialManifestMLS 
    output_manifest_file: ...
    download_dir: ...
    ...

  # use existing classes for common operations or write your own
  - _target_: sdp.processors.SubSubstringToSubstring 

    substring_pairs: { 
      # specify the parameters needed for your usecase 
      " mr ": " mister ",
      " misteak ": " mistake ",
      ...
    }

  - _target_: sdp.processors.DropNonAlphabet 
    alphabet: " abcdefghijklmnopqrstuvwxyz"
    output_manifest_file: ... 
    ...
```
## Existing processor classes
In addition to those mentioned in the example config file, many more classes are already included in Speech Data Processor, for example:
* `sdp.processors.ASRInference` will run inference on the manifest using a specified `pretrained_model`.
* `sdp.processors.DropHighWER` will compute WER between `text` and `pred_text` of each utterance and remove the utterance if WER is greater than the specified `wer_threshold`.
* `sdp.processors.DropHighLowCharrate` will compute the character rate in the utterance using `text` and `duration`, and drop the utterance if it is outside the bounds of the specified `high_charrate_threshold` and `low_charrate_threshold`. Carefully chosen thresholds will allow us to drop utterances with incorrect ground truth `text`.

## Processor test cases
You can add test cases to verify you have specified your desired changes correctly and to help document why your are making these changes.

For example:
```yaml
processors:
  ...
  - _target_: sdp.processors.DropIfRegexInAttribute
    attribute_to_regex:
      "text" : ["(\\D ){5,20}"] # looks for between 4 and 19 characters surrounded by spaces

    test_cases:
      - {input: {text: "some s p a c e d out letters"}, output: null}
      - {input: {text: "normal words only"}, output: {text: "normal words only"}}
      - {input: {text: "three a b c spaced out letters"}, output: {text: "three a b c spaced out letters"}}
      - {input: {text: "four a b c d spaced out letters"}, output: null}
  ...
```