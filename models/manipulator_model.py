import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=1, r3=0.05):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 3.
        self.l2 = 0.4   
        self.r2 = 0.01
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2
        self.alpha = self.I_1 + self.I_2 + self.m1 * self.d1 ** 2 + self.m2 * (self.l1 ** 2 + self.d2 ** 2) + \
                self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        self.beta = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        self.delta = self.I_2 + self.m2 * self.d2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        m_11 = self.alpha + 2 * self.beta * np.cos(q2)
        m_12 = self.delta + self.beta * np.cos(q2)
        m_21 = m_12
        m_22 = self.delta
        return np.array([[m_11, m_12], [m_21, m_22]])
    
    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x
        c_11 = -self.beta * np.sin(q2) * q2_dot
        c_12 = -self.beta * np.sin(q2) * (q1_dot + q2_dot)
        c_21 = self.beta * np.sin(q2) * q1_dot
        c_22 = 0
        return np.array([[c_11, c_12], [c_21, c_22]])
