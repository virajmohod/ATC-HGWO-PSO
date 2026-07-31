import numpy as np

from algorithms.proposed import EliteArchive

archive = EliteArchive(

    archive_size=5,

)

for i in range(15):

    mask = np.random.randint(

        0,

        2,

        20,

    )

    archive.add(

        mask,

        np.random.rand(),

    )

print()

print(archive.statistics())

print()

print(archive.best())

print()

print(archive.mean_position())