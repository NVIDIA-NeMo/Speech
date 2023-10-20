Self-Supervised Learning
=================================

Self-Supervised Learning (SSL) refers to the problem of learning without explicit labels. As 
any learning process require feedback, without explit labels, SSL derives supervisory signals from 
the data itself. The general ideal of SSL is to predict any hidden part (or property) of the input 
from observed part of the input (e.g., filling in the blanks in a sentence or predicting whether 
an image is upright or inverted).

SSL for speech/audio understanding broadly falls into either contrastive or reconstruction 
based approaches. In contrastive methods, models learn by distinguishing between true and distractor 
tokens (or latents). Examples of contrastive approaches are Contrastive Predictive Coding (CPC), 
Masked Language Modeling (MLM) etc. In reconstruction methods, models learn by directly estimating 
the missing (intentionally leftout) portions of the input. Masked Reconstruction, Autoregressive 
Predictive Coding (APC) are few examples.

In the recent past, SSL has been a major benefactor in improving Acoustic Modeling (AM), i.e., the 
encoder module of neural ASR models. Here too, majority of SSL effort is focused on improving AM. 
While it is common that AM is the focus of SSL in ASR, it can also be utilized in improving other parts of 
ASR models (e.g., predictor module in transducer based ASR models).

The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   results
   configs
   api
   resources

.. include:: resources.rst
