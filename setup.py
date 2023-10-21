from setuptools import setup

setup(
    name         = "NeuralCompression_v2",
    version      = "0.0.1.dev",
    author       = "Yi Huang",
    author_email = "yhuang2@bnl.gov",
    description  = ("Time-projection chamber data compression "
                   "with Bicephalous Convolutional Autoencoder"),
    license      = "BSD 3-Clause 'New' or 'Revised' License",
    keywords     = "autoencoder, data compression, high-energy physics",
    packages     = ['neuralcompress_v2'],
    classifiers  = [
        "Development Status :: 3 - Alpha",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License",
    ],
)
