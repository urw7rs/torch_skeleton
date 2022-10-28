Transforms
==========

.. currentmodule:: torch_skeleton.transforms

Compositions of transforms
--------------------------

.. autosummary::

   Compose


Padding
-------

.. autosummary::

   PadBodies
   PadFrames

Selection
---------

.. autosummary::

   SampleFrames
   SelectKBodies

Splitting
---------

.. autosummary::

   SplitFrames

Normalization
-------------

.. autosummary::

   CenterJoint
   ParallelBone

Augmentation
------------

.. autosummary::

   RandomShift
   RandomRotate

Denoising
---------
.. autosummary::

   SortByMotion
   DenoiseByLength
   DenoiseBySpread
   DenoiseByMotion
   MergeBodies
   RemoveZeroFrames
   Denoise

.. automodule:: torch_skeleton.transforms
   :members:
