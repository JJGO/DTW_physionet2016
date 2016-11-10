from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

classifier_dict = {
    'LSVC': Pipeline([
        ('norm', MinMaxScaler(feature_range=(0, 1))),
        ('clf',  LinearSVC(dual=False, tol=1e-4))]),

    'LSVM': Pipeline([
        ('norm', MinMaxScaler(feature_range=(0, 1))),
        ('clf',  SVC(kernel='linear', class_weight='balanced'))]),

    'RSVM':   Pipeline([
        ('norm', MinMaxScaler(feature_range=(0, 1))),
        ('clf',  SVC(kernel='rbf', class_weight='balanced'))]),

    'LSVH': Pipeline([
        ('norm', MinMaxScaler(feature_range=(0, 1))),
        ('clf',  LinearSVC(loss='hinge', class_weight='balanced', tol=1e-4))]),

    'PSVM': Pipeline([
        ('norm', MinMaxScaler(feature_range=(0, 1))),
        ('clf',  SVC(kernel='poly', class_weight='balanced'))]),

    'RF': RandomForestClassifier(),
}
