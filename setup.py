from setuptools import setup
import codecs


with codecs.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="cmsuml",
    version=0.1,
    description="Unsupervised methods of machine learning in CMS",
    long_description=long_description,

    url='https://github.com/anaderi/cms_unsupervised_eval',

    # Author details
    author='Alex Rogozhnikov',
    author_email='axelr@yandex-team.ru',

    # Choose your license
    license='Yandex',
    packages=['cmsuml'],
    package_dir={'cmsuml': 'cmsuml'},
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: CERN, CMS, HEP',
        'Topic :: YDF :: Cern Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: YDF :: Yandex License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='machine learning, unsupervised learning, high energy physics ',

    # List run-time dependencies here. These will be installed by pip when your
    # project is installed.
    install_requires = [
        'ipython >= 2.1.0',
        'pyzmq >= 14.3.0',
        'matplotlib >= 1.4',
        'rootpy >= 0.7.1',
        'root_numpy >= 3.3.0',
        'pandas >= 0.14.0',
        'scikit-learn >= 0.15',
        'scipy >= 0.14.0',
        'numpy >= 1.8.1',
        'jinja2 >= 2.7.3',
    ],
)