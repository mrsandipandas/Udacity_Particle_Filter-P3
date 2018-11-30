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

#define EPSILON 0.00001

using namespace std;

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	// x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 200;
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    for (int i = 0; i < num_particles; i++) {

        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;
        
        particles.push_back(particle);
	}
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    
    // Creating normal distributions
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    // Calculate new state.
    for (int i = 0; i < num_particles; i++) {
        double theta = particles[i].theta;
        
        // When yaw is not changing, avoid devision by Zero
        if ( fabs(yaw_rate) < EPSILON ) { 
          particles[i].x += velocity * delta_t * cos( theta );
          particles[i].y += velocity * delta_t * sin( theta );
        } else {
          particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
          particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
          particles[i].theta += yaw_rate * delta_t;
        }

        // Adding noise.
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (size_t i = 0; i < observations.size(); i++) {
        // Initialize min distance as a really big number
        double minDistance = numeric_limits<double>::max();

        // Initialize
        int mapId = -1;

        for (size_t j = 0; j < predicted.size(); j++ ) { // For each predition.

            double xDistance = observations[i].x - predicted[j].x;
            double yDistance = observations[i].y - predicted[j].y;

            double distance = xDistance * xDistance + yDistance * yDistance;

            // Find the nearest landmark iteratively for each observation - O(N)
            if ( distance < minDistance ) {
                minDistance = distance;
                mapId = predicted[j].id;
            }
        }

        // Update the observation identifier.
        observations[i].id = mapId;
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
    
    for (int i = 0; i < num_particles; i++) {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        vector<LandmarkObs> predictions;

        for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++) {

            // get id and x,y coordinates
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
            int lm_id = map_landmarks.landmark_list[j].id_i;

            // only consider landmarks within sensor range of the particle
            if (dist(lm_x, lm_y, p_x, p_y) <= sensor_range) {
                predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
            }
        }

        // Convert observations from vehicle coordinates to map coordinates
        vector<LandmarkObs> observations_map_coordinates;
        for (size_t j = 0; j < observations.size(); j++) {
            double m_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
            double m_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
            observations_map_coordinates.push_back(LandmarkObs{ observations[j].id, m_x, m_y });
        }

        // perform dataAssociation for the predictions and transformed observations on current particle
        dataAssociation(predictions, observations_map_coordinates);

        // reinit weight
        particles[i].weight = 1.0;

        for (size_t j = 0; j < observations_map_coordinates.size(); j++) {

            double obs_x, obs_y, pr_x, pr_y;
            obs_x = observations_map_coordinates[j].x;
            obs_y = observations_map_coordinates[j].y;

            int associated_prediction_id = observations_map_coordinates[j].id;

            // get the x,y coordinates of the prediction associated with the current observation
            for (size_t k = 0; k < predictions.size(); k++) {
                if (predictions[k].id == associated_prediction_id) {
                    pr_x = predictions[k].x;
                    pr_y = predictions[k].y;
                }
            }

            // calculate weight with multivariate Gaussian
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            double obs_w = ( 1/(2*M_PI*std_x*std_y)) * exp(-( pow(pr_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(pr_y-obs_y,2)/(2*pow(std_y, 2)))));
            
            particles[i].weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<double> weights;
    double maxWeight = numeric_limits<double>::min();
    for(int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
        if ( particles[i].weight > maxWeight ) {
            maxWeight = particles[i].weight;
        }
    }

    // Creating distributions.
    uniform_real_distribution<double> distDouble(0.0, maxWeight);
    uniform_int_distribution<int> distInt(0, num_particles - 1);

    // Generating index.
    int index = distInt(gen);

    double beta = 0.0;

    // the wheel concept by Sebastian
    vector<Particle> resampledParticles;
    for(int i = 0; i < num_particles; i++) {
        beta += distDouble(gen) * 2.0;
        while( beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        resampledParticles.push_back(particles[index]);
    }

    particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
