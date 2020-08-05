% https://ch.mathworks.com/help/stats/examples/classification.html#d120e3660

clear all, close all , clc; 
% Measurements and priors (mean and stddev for each parameter)
gouyvertical = 46 * pi / 180; % Measured vertical Gouy phase in radian
gouyhorizontal = 35 * pi / 180; % Measured horizontal Gouy phase in radian
gouystd = 0.1;
dmean = .102; % Distance one
dstd = .05;
Lmean = 1.45; % Distance two
Lstd = .2;
Rmean = 100e-3; % ROC
Rstd = 20e-3;
Thetamean = 3.8 * pi / 180; % Angle of incidence in radian
Thetastd = 2 * pi / 180;

% Probability distribution function
gaussian = @(mean, std, x) 1 / sqrt(2 * pi * std^2) * exp( -(x - mean)^2 / (2 * std^2) );

% Model connecting parameters with measurements
gouyphasevert = @(L, R, theta, d) (R^2 - 2 * (d + L) * R * cos(theta) + 2 * d * L * cos(theta)^2) / R^2;
gouyphasehor = @(L, R, theta, d) (R^2 - 2 * (d + L) * R / cos(theta) + 2 * d * L / cos(theta)^2) / R^2;

% Negative log probability, to be minimised
lossfunction = @(xvert,xhor,params) -log(gaussian(acos(gouyphasevert(params(1), params(2), params(3), params(4))), gouystd, xvert) ...
    * gaussian(acos(gouyphasehor(params(1), params(2), params(3), params(4))), gouystd, xhor) ...
    * gaussian(Lmean, Lstd, params(1)) * gaussian(Rmean, Rstd, params(2)) * gaussian(Thetamean, Thetastd, params(3)) * gaussian(dmean, dstd, params(4)));
fit_error = @(params) lossfunction(gouyvertical, gouyhorizontal, params);

% Optimisation
start_params = [Lmean, Rmean, Thetamean, dmean];
fit_params = fminsearch(fit_error, start_params, optimset('PlotFcns', @optimplotfval))

% Convert results to degree & double-check whether estimated params result in measurement according to model
thetafit = 180 * fit_params(3) / pi
acos( gouyphasevert( fit_params(1), fit_params(2), fit_params(3), fit_params(4) ) ) * 180 / pi
acos( gouyphasehor( fit_params(1), fit_params(2), fit_params(3), fit_params(4) ) ) * 180 / pi