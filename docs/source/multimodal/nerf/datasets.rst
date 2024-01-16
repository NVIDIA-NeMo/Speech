Datasets
========

.. note:: It is the responsibility of each user to check the content of the dataset, review the applicable licenses, and determine if it is suitable for their intended use. Users should review any applicable links associated with the dataset before placing the data on their machine.


Rays dataset
------------
Ray datasets are specialized data structures designed for applications in computer graphics, notably in 3D reconstruction, neural rendering, and ray tracing.

Ray datasets are characterized by their detailed representation of rays, each defined by an origin point (rays_o) and a direction vector (rays_d).
These datasets are closely tied to specific image dimensions, including height and width, which dictate the resolution and aspect ratio of the target images.
Alongside the core ray data, these datasets typically include additional metadata such as camera parameters, depth values, and color information.
The diversity and complexity of the dataset, encompassing a range of viewpoints and lighting conditions, play a crucial role in capturing the nuances of real-world light behavior.


Random Poses Dataset
^^^^^^^^^^^^^^^^^^^^
The Random Poses Dataset randomly generates camera poses, each translating to a unique set of rays characterized by their origins and directions.
This randomization is key to covering a wide range of potential viewpoints and angles, mimicking a comprehensive exploration of a 3D scene.
This diverse sampling is essential for training robust NeRF models capable of accurately reconstructing and rendering 3D environments from previously unseen angles.

The dataset inherently accounts for the necessary parameters of ray generation, such as the height and width of the target images,
ensuring that the rays are compatible with the specific requirements of the rendering or reconstruction algorithms.
In addition to the ray origins and directions, the dataset may also include other relevant metadata like camera intrinsic and extrinsic parameters,
contributing to a more detailed and versatile training process.

An example of RandomPosesDataset usage as a training dataset is shown below:

.. code-block:: yaml

  model:
    data:
      train_batch_size: 1
      train_shuffle: false
      train_dataset:
        _target_: nemo.collections.multimodal.data.nerf.random_poses.RandomPosesDataset
        internal_batch_size: 100
        width: 512
        height: 512
        radius_range: [3.0, 3.5]
        theta_range: [45, 105]
        phi_range: [-180, 180]
        fovx_range: [10, 30]
        fovy_range: [10, 30]
        jitter: False
        jitter_center: 0.2
        jitter_target: 0.2
        jitter_up: 0.02
        uniform_sphere_rate: 0
        angle_overhead: 30
        angle_front: 60


Circle Poses Dataset
^^^^^^^^^^^^^^^^^^^^
Circle Poses Dataset is a specialized ray dataset designed for generating samples of rays in a circular pattern.
The key feature of this dataset is its ability to simulate camera positions arranged along a circular path, focusing on a central point.
This arrangement is particularly useful for capturing scenes from multiple, evenly spaced angles, ensuring a comprehensive view around a central axis.

The defining parameter of the Circle Poses Dataset is its size, which dictates the number of samples or camera poses around the circle.
A larger size results in more camera positions being generated, offering finer granularity and coverage of the circle.
Each camera pose corresponds to a unique set of rays, with origins and directions calculated based on the position around the circle and the focus on the central point.

The Circle Poses Dataset is particularly valuable during validation and testing to generate a holistic view of the reconstructed scene.

An example of CirclePosesDataset usage as a validation dataset is shown below:

.. code-block:: yaml

  model:
    data:
      val_batch_size: 1
      val_shuffle: false
      val_dataset:
        _target_: nemo.collections.multimodal.data.nerf.circle_poses.CirclePosesDataset
        size: 5
        width: 512
        height: 512
        angle_overhead: 30
        angle_front: 60
