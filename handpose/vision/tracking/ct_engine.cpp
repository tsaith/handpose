#include "../include/ct.h"

void c_get_feature_values_v1(
  vector<vector<double> > &feature_values,
  vector<vector<int> > boxes,
  vector<vector<vector<int> > > features,
  vector<vector<double> > weights,
  vector<vector<double> > integral_image){

  int n_boxes = boxes.size();
  int n_features = features.size();

  int n_rects;
  int i, j, k;

  int x_min, x_max;
  int y_min, y_max;
  double weight, value;

  int box_x, box_y;
  int rect_x, rect_y;
  int rect_width, rect_height;

  // Get the feature values
  for (i = 0; i < n_features; i++){
    n_rects = 4;
    for (j = 0; j < n_boxes; j++){
      value = 0.0;
      for (k = 0; k < n_rects; k++){
        box_x = boxes[j][0];
        box_y = boxes[j][1];

        rect_x = features[i][k][0];
        rect_y = features[i][k][1];
        rect_width  = features[i][k][2];
        rect_height = features[i][k][3];

        x_min = box_x + rect_x;
        y_min = box_y + rect_y;
        x_max = x_min + rect_width;
        y_max = y_min + rect_height;

        weight = weights[i][k];
        value += weight * (integral_image[y_min][x_min] +
                           integral_image[y_max][x_max] -
                           integral_image[y_min][x_max] -
                           integral_image[y_max][x_min]);

        feature_values[i][j] = value;

      }
    }
  }
}

void c_get_feature_values(
  vector<vector<double> > &feature_values,
  vector<Rect> boxes,
  vector<vector<Rect> > features,
  vector<vector<double> > weights,
  vector<vector<double> > integral_image){

  int n_boxes = boxes.size();
  int n_features = features[0].size();

  int n_rects;
  int i, j, k;

  int x_min, x_max;
  int y_min, y_max;
  double weight, value;

  int box_x, box_y;
  int rect_x, rect_y;
  int rect_width, rect_height;

  // Get the feature values
  for (i = 0; i < n_features; i++){
    n_rects = 4;
    for (j = 0; i < n_boxes; j++){
      value = 0.0;
      for (k = 0; k < n_rects; k++){
        box_x = boxes[j].x;
        box_y = boxes[j].y;

        rect_x = features[i][k].x;
        rect_y = features[i][k].y;
        rect_width  = features[i][k].width;
        rect_height = features[i][k].height;

        x_min = box_x + rect_x;
        y_min = box_y + rect_y;
        x_max = x_min + rect_width;
        y_max = y_min + rect_height;

        weight = weights[i][k];
        value += weight * (integral_image[y_min][x_min] +
                           integral_image[y_max][x_max] -
                           integral_image[y_min][x_max] -
                           integral_image[y_max][x_min]);

        feature_values[i][j] = value;

      }
    }
  }
}

void c_ratio_classifier(
    int &index_ratio_max, double &ratio_max,
    vector<double> mu_pos, vector<double> sigma_pos,
    vector<double> mu_neg, vector<double> sigma_neg,
    vector<vector<double> > feature_values) {


    double err_tol = 1.0e-30; // error tolerance

    ratio_max = -1.0e30;
    index_ratio_max = 0;

    int n_features = feature_values.size();
    int n_samples = feature_values[0].size();

    int i, j;
    double rsum, v;
    double fac, proba_pos, proba_neg;

    for (j = 0; j < n_samples; j++){
        rsum = 0.0;
        for (i = 0; i < n_features; i++){
            v = feature_values[i][j];

            fac = 1.0 / (sigma_pos[i] + err_tol);
            proba_pos = fac * exp(-0.5 * (v - mu_pos[i])*(v - mu_pos[i]) / (sigma_pos[i]*sigma_pos[i] + err_tol));

            fac = 1.0 / (sigma_neg[i] + err_tol);
            proba_neg = fac * exp(-0.5 * (v - mu_neg[i])*(v - mu_neg[i]) / (sigma_neg[i]*sigma_neg[i] + err_tol));

            rsum += log(proba_pos + err_tol) - log(proba_neg + err_tol);
        }

        if (ratio_max < rsum) {
            ratio_max = rsum;
            index_ratio_max = j;
        }
    }
}
