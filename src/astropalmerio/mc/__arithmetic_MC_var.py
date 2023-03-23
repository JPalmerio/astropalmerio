# NOTE: defining basic arithmetic operations for MC_var has proven too difficult
# I leave my best attempt at the code here for now but eventually it should be deleted
# def __add__(self, other):

#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self + real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value + float(other)
#         # Apply to bounds as well
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val + float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations + float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __sub__(self, other):

#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self - real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value - float(other)
#         # Apply to bounds as well
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val - float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations - float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __mul__(self, other):
#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self * real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value * float(other)
#         # Apply to bounds
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val * float(other)
#         # Apply to errors
#         for e, val in result.error.items():
#             if val is not None:
#                 result.error[e] = val * float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations * float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __div__(self, other):
#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_self / real_other
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         result = self.copy()
#         result.value = result.value / float(other)
#         # Apply to bounds
#         for b, val in result.bounds.items():
#             if val is not None:
#                 result.bounds[b] = val / float(other)
#         # Apply to errors
#         for e, val in result.error.items():
#             if val is not None:
#                 result.error[e] = val / float(other)
#         # And to realizations if they exist
#         if result.realizations is not None:
#             result.realizations = result.realizations / float(other)
#         return result
#     else:
#         raise TypeError("Unsupported operand type")

# def __pow__(self, other):
#     if type(other) == type(self):

#         if any(other.lim.values()) or any(self.lim.values()):
#             raise NotImplementedError("Cannot perform this operation on MC_var objects with limits.")

#         N_MC = self._check_sample_sizes(other)
#         self._check_seeds(other)

#         if self.realizations is None:
#             real_self = self.sample(N_MC=N_MC)
#         else:
#             real_self = self.realizations[:N_MC]

#         if other.realizations is None:
#             real_other = other.sample(N_MC=N_MC)
#         else:
#             real_other = other.realizations[:N_MC]

#         real_result = real_other ** real_self
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     elif isinstance(other, (float, int)):
#         real_result = other ** real_self
#         value = np.quantile(real_result, 0.5)
#         errp = value - np.quantile(real_result, 0.16)
#         errm = np.quantile(real_result, 0.84) - value
#         result = MC_var(value, error=[errm, errp], N_MC=N_MC)
#         result.realizations = real_result
#         return result

#     else:
#         raise TypeError("Unsupported operand type")

# def log10(self):

#     if self.seed is not None:
#         np.random.seed(self.seed)
#     draw = self.asym_gaussian_draw(
#         self.value, self.errm, self.errp, nb_draws=self.N_MC
#     )
#     result = np.log10(draw)
#     q = stats.mstats.mquantiles(result, prob=[0.16, 0.5, 0.84])
#     value = q[1]
#     errp = q[2] - q[1]
#     errm = q[1] - q[0]
#     return MC_var(value, errp, errm, N=self.N_MC)

# def exp(self):

#     if self.seed is not None:
#         np.random.seed(self.seed)
#     draw = self.asym_gaussian_draw(
#         self.value, self.errm, self.errp, nb_draws=self.N_MC
#     )
#     result = np.exp(draw)
#     q = stats.mstats.mquantiles(result, prob=[0.16, 0.5, 0.84])
#     value = q[1]
#     errp = q[2] - q[1]
#     errm = q[1] - q[0]
#     return MC_var(value, errp, errm, N=self.N_MC)

# def _check_sample_sizes(self, other):
#     if other.N_MC != self.N_MC:
#         log.warning(
#             "Trying to perform an operation on two MC_var objects with different number of samples, "
#             "falling back to the smallest sample size."
#         )
#         newN = min(other.N_MC, self.N_MC)
#     else:
#         newN = self.N_MC
#     return newN

# def _check_seeds(self, other):
#     if (other.seed == self.seed) and (self.seed is not None):
#         log.warning(
#             "Performing an operation on two MC_var objects with the same seed "
#             "will result in correlated realizations."
#         )
