from setuptools import find_packages, setup


if __name__ == '__main__':
    setup(
        name='LiSeer',
        version='0.0',
        description=("LiSeer: zero-shot LiDAR semantic segmentation via scene-sensor disentanglement"),
        author='LiSeer Contributors',
        author_email='xuzk23@mails.tsinghua.edu.cn',
        keywords='LiSeer LiDAR Semantic Segmentation',
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="Apache License 2.0",
    )
