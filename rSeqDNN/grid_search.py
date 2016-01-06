import ast
import types
from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint

def is_valid_python(code):
   try:
       ast.parse(code)
   except SyntaxError:
       return False
   return True

def find_method(obj, method):
    try:
        if isinstance(method, str):
            return getattr(obj, method)
        else:
            return types.MethodType(method, obj)
    except AttributeError as e:
        print method, 'method not found for object ', obj
        raise e

def send_email(the_subject, the_to, the_from="av-mail-sender@stanford.edu", the_contents=""):
   '''copied from av_scripts
   '''
   import smtplib
   from email.mime.text import MIMEText
   msg = MIMEText(the_contents)
   msg['Subject'] = the_subject
   msg['From'] = the_from
   msg['To'] = ",".join(the_to)
   s = smtplib.SMTP('smtp.stanford.edu')
   s.starttls()
   s.sendmail(the_from, the_to, msg.as_string())
   s.quit()

class MOESearch(object):
    def __init__(self, estimator, param_grid, param_types,
                 fixed_param=None, conditional_param=None):
        """
        MOE hyper parameter search.
        Limited to continuous and discrete parameters only.

        Parameters
        ----------
        estimator : estimator classobj
            estimator needs to provide fit and score methods.
        param_grid : dict
            Dictionary with parameter names as keys and lists of
            lower and upper bounds to search over as values.
        param_types : dict
            Dictionary with parameter names as keys and types as
            values. Legal values: 'cont', 'disc'.
        fixed_param : dict, optional
            Parameters that remain fixed during the search.
        conditional_param : dict, optional
            Parameters conditional on the param_grid.
            For example, {maxpool_stride: 'maxpool_size/2'}.

        Attributes
        ----------
        experiment : MOE experiment
        grid_scores_ : list
        grid_params_ : list of dicts
        best_score_ : float
        best_grid_params_ : dict
        best_estimator_ : estimator
        """
        self.estimator = estimator
        assert all(len(value)==2 and value[1]>value[0] for key, value
                   in param_grid.iteritems() if param_types[key] in ['cont', 'disc']), \
        "Invalid parameter grid!"
        self.param_grid = param_grid
        assert all(value in ['cont', 'disc'] for key, value in param_types.iteritems()), \
        "Invalid parameter types!"
        self.param_types = param_types
        self.fixed_param = fixed_param if fixed_param is not None else {}
        self.conditional_param = conditional_param if conditional_param is not None else {}
        # start MOE experiment
        self.experiment = Experiment(self.param_grid.values())
        self.grid_scores_ = []
        self.grid_params_ = []
        self.best_score_ = None
        self.best_grid_param_ = {}
        self.best_estimator_ = None

    def eval_conditional_param(self, curr_grid_param):
        """
        Returns evaluated conditional parameters.

        Parameters
        ----------
        curr_param_grid : dict
            param_grid names as keys and values as values.

        Returns
        -------
        curr_conditional_param : dict
        """
        curr_conditional_param = self.conditional_param.copy()
        for key, value in curr_conditional_param.iteritems():
            for name in curr_grid_param.keys():
                if name in value:
                    value = value.replace(name, str(curr_grid_param[name]))
            if is_valid_python(value):
                curr_conditional_param[key] = eval(value)

        return curr_conditional_param

    def get_estimator_param(self, curr_grid_param):
        """
        Returns full dictionary of estimator parameters.

        Parameters
        ----------
        curr_grid_param : dict

        Returns
        -------
        estimator_param : dict
        """
        estimator_param = self.fixed_param.copy()
        estimator_param.update(curr_grid_param)
        curr_conditional_param = self.eval_conditional_param(curr_grid_param)
        estimator_param.update(curr_conditional_param)

        return estimator_param

    def fit(self, fit_method, fit_param, score_method, score_param,
            minimize=False, max_iter=2, email_updates_to=None):
        """
        Runs MOE search over the parameter grid.

        Parameters
        ----------
        fit_method : string or callable
        fit_param : dict
            Dictionary with parameter names as keys and
            parameter values as values to pass to fit_method
        score_method : string or callable
        score_param : dict
            Dictionary with parameter names as keys and
            parameter values as values to pass to score_method
        minimize : boolean
            Minimizes score if true, otherwise maximizes it.
        max_iter : int, default=2
            Maximum number of search iterations.

        Returns
        -------
        self
        """
        for i in xrange(max_iter):
            # sample next point, get estimator parameters
            next_point_to_sample = [val if self.param_types.keys()[indx]=='cont' else int(round(val))\
                                    for indx, val in enumerate(gp_next_points(self.experiment)[0])]
            curr_grid_param = dict(zip(self.param_grid.keys(), next_point_to_sample))
            estimator_param = self.get_estimator_param(curr_grid_param)
            # initialize estimators with new parameters
            model = self.estimator(**estimator_param)
            # fit, score, store results
            fit = find_method(model, fit_method)
            fit_model = fit(**fit_param)
            score = find_method(fit_model, score_method)
            value_of_next_point = score(**score_param)
            if not minimize:
               self.experiment.historical_data.append_sample_points(
                  [SamplePoint(next_point_to_sample, -value_of_next_point)])
            else:
               self.experiment.historical_data.append_sample_points(
                  [SamplePoint(next_point_to_sample, value_of_next_point)])
            self.grid_scores_.append(value_of_next_point)
            self.grid_params_.append(curr_grid_param)
            if (not minimize and value_of_next_point > self.best_score_)\
                or (minimize and value_of_next_point < self.best_score_):
                if email_updates_to is not None:
                   subject = 'grid search %s update' % self.estimator.__name__
                   contents = '\n'.join(['iteration %i of %i' % (i, max_iter),
                                         'new best score: '+str(value_of_next_point),
                                         'previous best score: '+str(self.best_score_)])
                   send_email(subject, email_updates_to, the_contents=contents)
                self.best_score_ = value_of_next_point
                self.best_grid_param_ = curr_grid_param
                self.best_estimator_ = fit_model

        return self
