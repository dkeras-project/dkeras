Why Use dKeras?
---------------

Distributed deep learning can be essential for production systems where you
need fast inference but don't want expensive hardware accelerators or when
researchers need to train large models made up of distributable parts.

This becomes a challenge for developers because they'll need expertise in not
only deep learning but also distributed systems. A production team might also
need a machine learning optimization engineer to use neural network
optimizers in terms of precision changes, layer fusing, or other techniques.

Distributed inference is a simple way to get better inference FPS. The graph
below shows how non-optimized, out-of-box models from default frameworks can
be quickly sped up through data parallelism: