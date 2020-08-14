from setuptools import setup

setup(
   name='flatlander',
   version='0.1',
   description='Contains flatland challenge solutions',
   author='Pascal Wullschleger',
   author_email='pascal.wullschleger@hslu.ch',
   packages=['flatlander'],
   entry_points={
      "console_scripts": ["flatlander = flatlander.scripts.__main__:main"]
   }
)