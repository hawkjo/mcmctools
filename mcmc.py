import sys
import numpy as np


class MCMCSampleCollection(object):
    """
    The MCMCSampleCollection is simply a way to organize MCMC runs conveniently and consistently,
    with appropriate methods for convergence testing built-in.
    """
    # ---------------------------------------------------------------------------------------------
    # Overview of MCMCSampleCollection:
    #
    # At its core, this class is simply a convenient way to store 'sample tuples'. A sample tuple
    # is of the form:
    #                   (sample index, variable index, variable value)
    #
    # A sample is simply any collection of sample tuples which have the same sample index.
    #
    # One step further, this allows us a convenient way to standardize convergence tests.
    # ---------------------------------------------------------------------------------------------

    def __init__(self):
        self.num_vars = 0
        self.var_index_given_name = {}
        self.var_name_given_index = {}
        self.num_samples = 1
        self.sample_tuples = []

    def init_var(self, var_name, value):
        if self.num_samples > 1:
            sys.exit('Attempting to initialize variable after sampling has begun.')
        var_ind = self.num_vars
        self.var_index_given_name[var_name] = var_ind
        self.var_name_given_index[var_ind] = var_name
        self.add_sample_by_index(var_ind, value, same_sample_as_prev=True)
        self.num_vars += 1

    def add_sample_by_index(self, var_ind, value, same_sample_as_prev=False):
        if not same_sample_as_prev:
            self.num_samples += 1
        sample_ind = self.num_samples - 1
        self.sample_tuples.append((sample_ind, var_ind, value))

    def add_sample_by_name(self, var_name, value, same_sample_as_prev=False):
        self.add_sample_by_index(self.var_index_given_name[var_name], value, same_sample_as_prev)

    def first_tuple_index_of_desired_sample_index(self, desired_sample_ind):
        assert desired_sample_ind <= self.num_samples - 1, \
            'Desired index %d is out of bounds with number of samples %d' \
            % (desired_sample_ind, self.num_samples)

        if desired_sample_ind <= 10:
            for i, sample_ind in enumerate(self.sample_tuples):
                if sample_ind == desired_sample_ind:
                    return i
            else:
                sys.exit('Cannot find sample index %d. Total of %d samples in collection.'
                         % (desired_sample_ind, self.num_samples))
        else:
            # Binary search
            lower_bound = 0
            upper_bound = len(self.sample_tuples)
            while True:
                # Find index such that previous sample_ind < desired_sample_ind
                #   and current sample_ind == desired_sample_ind
                # Note that we know test_ind-1 > 0 because we already dealt with sample_ind <= 10.
                # Also note that test_ind < len(self.sample_tuples) as integer division rounds
                # down.
                test_ind = (upper_bound + lower_bound)/2
                if self.sample_tuples[test_ind-1][0] >= desired_sample_ind:
                    upper_bound = test_ind
                elif self.sample_tuples[test_ind][0] < desired_sample_ind:
                    lower_bound = test_ind
                else:
                    return test_ind
                # We must have lower_bound + 1 < upper_bound. Otherwise we will have tested both
                # lower_bound and upper_bound right next to each other or out of order, but not
                # found our desired index.
                assert lower_bound + 1 < upper_bound

    def full_posterior_samples(self, starting_sample_ind=0, ending_sample_ind=None):
        if ending_sample_ind is None:
            ending_sample_ind = self.num_samples
        posterior_samples = np.zeros((ending_sample_ind - starting_sample_ind, self.num_vars))
        current_sample_ind = 0
        current_sample = np.zeros((self.num_vars))
        for sample_ind, var_ind, value in self.sample_tuples:
            if current_sample_ind != sample_ind:
                assert current_sample_ind > sample_ind
                if current_sample_ind >= starting_sample_ind:
                    posterior_samples[current_sample_ind - starting_sample_ind] = current_sample
                current_sample_ind = sample_ind
                if current_sample_ind >= ending_sample_ind:
                    break
            current_sample[var_ind] = value
        return posterior_samples

    def marginal_posterior_samples_by_name(self,
                                           marg_var_names,
                                           starting_sample_ind=0,
                                           ending_sample_ind=None):
        if ending_sample_ind is None:
            ending_sample_ind = self.num_samples
        marg_var_inds = [self.var_index_given_name[name] for name in marg_var_names]
        self.marginal_posterior_samples_by_index(marg_var_inds,
                                                 starting_sample_ind=0,
                                                 ending_sample_ind=self.num_samples)

    def marginal_posterior_samples_by_index(self,
                                            marg_var_inds,
                                            starting_sample_ind=0,
                                            ending_sample_ind=None):
        if ending_sample_ind is None:
            ending_sample_ind = self.num_samples
        posterior_samples = np.zeros((ending_sample_ind - starting_sample_ind, len(marg_var_inds)))
        current_sample_ind = 0
        current_sample = np.zeros((len(marg_var_inds)))
        for sample_ind, var_ind, value in self.sample_tuples:
            if current_sample_ind != sample_ind:
                assert current_sample_ind > sample_ind
                if current_sample_ind >= starting_sample_ind:
                    posterior_samples[current_sample_ind - starting_sample_ind] = current_sample
                current_sample_ind = sample_ind
                if current_sample_ind >= ending_sample_ind:
                    break
            current_sample[marg_var_inds.index(var_ind)] = value
        return posterior_samples
