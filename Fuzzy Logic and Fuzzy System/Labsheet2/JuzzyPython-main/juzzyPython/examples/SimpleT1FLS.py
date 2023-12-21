"""
SimpleT1FLS.py
Created 19/12/2021
"""
import math
import time
from juzzyPython.generic.Tuple import Tuple
from juzzyPython.generic.Output import Output
from juzzyPython.generic.Input import Input
from juzzyPython.generic.Plot import Plot
from juzzyPython.type1.system.T1_Rule import T1_Rule
from juzzyPython.type1.system.T1_Antecedent import T1_Antecedent
from juzzyPython.type1.system.T1_Consequent import T1_Consequent
from juzzyPython.type1.system.T1_Rulebase import T1_Rulebase
from juzzyPython.type1.sets.T1MF_Gaussian import T1MF_Gaussian
from juzzyPython.type1.sets.T1MF_Triangular import T1MF_Triangular
from juzzyPython.type1.sets.T1MF_Trapezoidal import T1MF_Trapezoidal
from juzzyPython.type1.sets.T1MF_Gauangle import T1MF_Gauangle
from juzzyPython.testing.timeRecorder import timeDecorator
class SimpleT1FLS:
    """
    Class SimpleT1FLS: 
    A simple example of a type-1 FLS based on the "How much to tip the waiter"  scenario.
    We have two inputs: food quality and service level and as an output we would
    like to generate the applicable tip.

    Parameters:None
        
    Functions: 
        getTip
        PlotMFs
        getControlSurfaceData
        
    """
    def __init__(self,unit = False) -> None:

        #Inputs to the FLS---temperature,headache,age
        self.temperature = Input("temperature", Tuple(0, 60))
        self.headache = Input("headache", Tuple(0, 10))
        self.age = Input("age", Tuple(0, 130))
        #Output
        self.urgency = Output("urgency", Tuple(0, 100))

        self.plot = Plot()

        #Set up the membership functions (MFs) for each input and output
        temperature_low_MF = T1MF_Triangular("MF for temperature_low", 0.0, 18.5, 36.5)
        temperature_normal_MF = T1MF_Triangular("MF for temperature_normal", 35.5, 36.6, 38.0)
        temperature_high_MF = T1MF_Triangular("MF for temperature_high", 37.5, 50, 60.0)

        headache_uncomfortable_MF = T1MF_Gaussian("MF for headache_uncomfortable", 3.0, 2.0)
        headache_heavy_MF = T1MF_Gaussian("MF for headache_heavy", 6.0, 2.0)

        age_young_MF = T1MF_Gauangle("MF for age_young", 0.0, 33.0, 65.0)
        age_old_MF = T1MF_Gauangle("MF for age_old", 55.0, 93.0, 130.0)

        not_urgent_MF = T1MF_Gauangle("Not urgent", 0.0, 20.0, 40.0)
        urgent_MF = T1MF_Gauangle("Urgent", 27.0, 50.0, 72.0)
        very_urgent_MF = T1MF_Gauangle("Very urgent", 60.0, 80.0, 100.0)

        #Set up the antecedents and consequents
        temperature_low = T1_Antecedent(temperature_low_MF, self.temperature, "temperature low")
        temperature_normal = T1_Antecedent(temperature_normal_MF, self.temperature, "temperature normal")
        temperature_high = T1_Antecedent(temperature_high_MF, self.temperature, "temperature high")

        headache_uncomfortable = T1_Antecedent(headache_uncomfortable_MF, self.headache, "headache uncomfortable")
        headache_heavy = T1_Antecedent(headache_heavy_MF, self.headache, "headache heavy")

        age_young = T1_Antecedent(age_young_MF, self.age, "age young")
        age_old = T1_Antecedent(age_old_MF, self.age, "age old")

        not_urgent = T1_Consequent(not_urgent_MF, self.urgency, "not urgent")
        urgent = T1_Consequent(urgent_MF, self.urgency, "urgent")
        very_urgent = T1_Consequent(very_urgent_MF, self.urgency, "very urgent")

        #Set up the rulebase and add rules
        self.rulebase = T1_Rulebase()
        self.rulebase.addRule(T1_Rule([temperature_low, headache_uncomfortable, age_young], consequent=urgent))
        self.rulebase.addRule(T1_Rule([temperature_low, headache_uncomfortable, age_old], consequent=urgent))
        self.rulebase.addRule(T1_Rule([temperature_low, headache_heavy, age_young], consequent=urgent))
        self.rulebase.addRule(T1_Rule([temperature_low, headache_heavy, age_old], consequent=very_urgent))
        self.rulebase.addRule(T1_Rule([temperature_normal, headache_uncomfortable, age_young], consequent=not_urgent))
        self.rulebase.addRule(T1_Rule([temperature_normal, headache_uncomfortable, age_old], consequent=not_urgent))
        self.rulebase.addRule(T1_Rule([temperature_normal, headache_heavy, age_young], consequent=urgent))
        self.rulebase.addRule(T1_Rule([temperature_normal, headache_heavy, age_old], consequent=very_urgent))
        self.rulebase.addRule(T1_Rule([temperature_high, headache_uncomfortable, age_young], consequent=urgent))
        self.rulebase.addRule(T1_Rule([temperature_high, headache_uncomfortable, age_old], consequent=very_urgent))
        self.rulebase.addRule(T1_Rule([temperature_high, headache_heavy, age_young], consequent=very_urgent))
        self.rulebase.addRule(T1_Rule([temperature_high, headache_heavy, age_old], consequent=very_urgent))

        #just an example of setting the discretisation level of an output - the usual level is 100
        self.urgency.setDiscretisationLevel(100)

        #get some outputs
        self.getTip(39, 5, 65)

        print(self.rulebase.toString())
        #Plot control surface, false for height defuzzification, true for centroid defuzz.
        # self.getControlSurfaceData(True, 100, 100)
        self.plotMFs("Temperature Membership Functions", [temperature_low_MF, temperature_normal_MF, temperature_high_MF], self.temperature.getDomain(), 100)
        self.plotMFs("Headache Membership Functions", [headache_uncomfortable_MF, headache_heavy_MF], self.headache.getDomain(), 100)
        self.plotMFs("Age Membership Functions", [age_young_MF, age_old_MF], self.age.getDomain(), 100)
        self.plotMFs("Urgency", [not_urgent_MF, urgent_MF, very_urgent_MF], self.urgency.getDomain(), 100)

        if not unit:
            self.plot.show()

    @timeDecorator
    def getTip(self, temperature, headache, age) -> None:
        """Calculate the output based on the two inputs"""
        self.temperature.setInput(temperature)
        self.headache.setInput(headache)
        self.age.setInput(age)
        print("The temperature was: "+str(self.temperature.getInput()))
        print("The headache was: "+str(self.headache.getInput()))
        print("The age was: "+str(self.age.getInput()))
        print("Using height defuzzification, the FLS recommends"
                + "the urgency is: "+str(self.rulebase.evaluate(0)[self.urgency]))
        print("Using centroid defuzzification, the FLS recommends"
                + "the urgency is: "+str(self.rulebase.evaluate(1)[self.urgency]))

    @timeDecorator
    # def getControlSurfaceData(self,useCentroidDefuzz,input1Discs,input2Discs,unit = False) -> None:
    #     """Get the data to plot the control surface"""
    #     if unit:
    #         test = []
    #     incrX = self.left_sensor.getDomain().getSize()/(input1Discs-1.0)
    #     incrY = self.right_sensor.getDomain().getSize()/(input2Discs-1.0)
    #     x = []
    #     y = []
    #     z = [ [0]*input1Discs for i in range(input2Discs)]
    #
    #     for i in range(input1Discs):
    #         x.append(i*incrX)
    #     for i in range(input2Discs):
    #         y.append(i*incrY)
    #
    #     for x_ in range(input1Discs):
    #         self.left_sensor.setInput(x[x_])
    #         for y_ in range(input2Discs):
    #             self.right_sensor.setInput(y[y_])
    #             if useCentroidDefuzz:
    #                 out = self.rulebase.evaluate(1).get(self.direction)
    #             else:
    #                 out = self.rulebase.evaluate(0).get(self.direction)
    #             if out == None or math.isnan(out):
    #                 z[y_][x_] = 0.0
    #                 if unit:
    #                     test.append(0.0)
    #             else:
    #                 z[y_][x_] = out
    #                 if unit:
    #                     test.append(out)
    #     if unit:
    #         return test
    #     self.plot.plotControlSurface(x,y,z,self.left_sensor.getName(),self.right_sensor.getName(),self.direction.getName())
        
    @timeDecorator
    def plotMFs(self, name, sets, xAxisRange, discretizationLevel):
        """Plot the lines for each membership function of the sets"""
        self.plot.figure()
        self.plot.title(name)
        for i in range(len(sets)):
            self.plot.plotMF(name.replace("Membership Functions",""), sets[i].getName(), sets[i], discretizationLevel, xAxisRange, Tuple(0.0, 1.0), False)
        self.plot.legend()
    
    

if __name__ == "__main__":
    SimpleT1FLS()
