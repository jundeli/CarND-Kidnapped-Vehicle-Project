/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	std::default_random_engine gen;

	std::normal_distribution<double> N_x(x, std[0]);
	std::normal_distribution<double> N_y(y, std[1]);
	std::normal_distribution<double> N_theta(theta, std[2]);

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(particle.weight);

		is_initialized = true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;

	// Loop over all particles
	for (unsigned int i = 0; i << num_particles; ++i) {
		double new_x;
		double new_y;
		double new_theta;

		if (yaw_rate == 0) {
			new_x = particles[i].x + velocity * delta_t*cos(particles[i].theta);
			new_y = particles[i].y + velocity * delta_t*sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else {
			new_x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + delta_t * yaw_rate));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		}

		std::normal_distribution<double> N_x(new_x, std_pos[0]);
		std::normal_distribution<double> N_y(new_y, std_pos[1]);
		std::normal_distribution<double> N_theta(new_theta, std_pos[2]);

		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Loop over all observations
	for (unsigned int o = 0; o < observations.size(); ++o) {

		// Initialize min distance by a very large number
		double min_distance = numeric_limits<double>::max();
		
		// Loop over all predictions for association
		for (unsigned int p = 0; p < predicted.size(); ++p) {
			double distance = sqrt(pow((predicted[p].x - observations[o].x), 2) + pow((predicted[p].y - observations[o].y), 2));
			
			// Find nearest neighbour
			if (distance < min_distance) {
				observations[o].id = predicted[p].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	// Loop over all particles
	for (int p = 0; p < num_particles; ++p) {

		// Declare transformed observations
		std::vector<LandmarkObs> trans_observations;
		for (unsigned int i = 0; i < observations.size(); ++i) {
			LandmarkObs trans_obs;

			// Perform the transformation from car to map coordinates
			trans_obs.x = particles[p].x + cos(particles[p].theta)*observations[i].x - sin(particles[p].theta)*observations[i].y;
			trans_obs.y = particles[p].y + sin(particles[p].theta)*observations[i].x + cos(particles[p].theta)*observations[i].y;
			trans_observations.push_back(trans_obs);
		}

		std::vector<LandmarkObs> pred_landmarks;
		for (unsigned int l = 0; l < map_landmarks.landmark_list.size(); ++l) {

			// predicted landmark in range
			LandmarkObs pred_landmark;

			double distance = sqrt(pow((particles[p].x - map_landmarks.landmark_list[l].x_f), 2) + pow((particles[p].y - map_landmarks.landmark_list[l].y_f), 2));
			if (distance < sensor_range) {
				pred_landmark.id = map_landmarks.landmark_list[l].id_i;
				pred_landmark.x = map_landmarks.landmark_list[l].x_f;
				pred_landmark.y = map_landmarks.landmark_list[l].y_f;

				// Assign landmarks in range to pred_landmarks
				pred_landmarks.push_back(pred_landmark);
			}
		}

		// Associate each transformed observation with a landmark identifier
		ParticleFilter::dataAssociation(pred_landmarks, trans_observations);

		// Declare associations, sense_x, and sense_y for each particle
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;

		// Initialize particle weight
		particles[p].weight = 1.0;

		// Calculate multivariate-Gaussian probability for each observation
		for (unsigned int i = 0; i < trans_observations.size(); ++i) {

			// Calaulate normalizer
			double normalizer = 1 / (2 * M_PI*std_landmark[0] * std_landmark[1]);
			double ux = pred_landmarks[trans_observations[i].id].x;
			double uy = pred_landmarks[trans_observations[i].id].y;
			double obs_prob = normalizer * exp(-.5*(pow((trans_observations[i].x - ux) / std_landmark[0], 2) + pow((trans_observations[i].y - uy) / std_landmark[1], 2)));

			// Calculate the particle's final weight
			particles[p].weight *= obs_prob;

			// Record particle's  association and map coordinates
			associations.push_back(trans_observations[i].id);
			sense_x.push_back(trans_observations[i].x);
			sense_y.push_back(trans_observations[i].y);
		}

		// Set particles and weights
		particles[p] = SetAssociations(particles[p], associations, sense_x, sense_y);
		weights[p] = particles[p].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::default_random_engine gen;
	std::discrete_distribution<int> distribution(weights.begin(), weights.end());
		
	std::vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; ++i) {
		resample_particles.push_back(particles[distribution(gen)]);
	}

	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	// Clear previous associations and map coordinates
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
