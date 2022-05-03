"""
.. module:: main

main
******

:Description: Main program

    Executes all the steps of the project to analyze the data and try to predict the target feature.

:Authors:
    benjami parellada

:Version: 

:Date:  
"""

__author__ = 'benjami parellada'

seed = 1984

import exploration
import modeling
import results

if __name__ == '__main__':
    exploration.main()
    modeling.main()
    results.main()
