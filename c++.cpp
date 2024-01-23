#include <iostream>
#include <vector>
#include <climits>
using namespace std;

pair<int, int> getTravelDestination(const vector<int>& ratings, int S, int M) {
    int S = input1;
    int S = input2;
    int numCountries = ratings.size() / S; // Calculate the number of countries
    int minRating = INT_MAX; // Initialize the minimum rating with maximum possible value
    int countryIndex = 0;
    
    // Find the country with the lowest rating
    for (int i = 0; i < numCountries; i++) {
        if (ratings[i * S] < minRating) {
            minRating = ratings[i * S];
            countryIndex = i;
        }
    }
    
    // Calculate the state index within the selected country based on the month
    int stateIndex = (M - 1) % S;
    
    return make_pair(countryIndex + 1, stateIndex + 1);
}

int main() {
    int N, S, M;
    cout << "Enter the length of the rating list (N): ";
    cin >> N;
    cout << "Enter the number of states per country (S): ";
    cin >> S;
    cout << "Enter the month (M): ";
    cin >> M;

    vector<int> ratings(N);
    cout << "Enter the ratings for each state: ";
    for (int i = 0; i < N; i++) {
        cin >> ratings[i];
    }

    pair<int, int> travelDestination = getTravelDestination(ratings, S, M);
    cout << "Tyler will travel to Country " << travelDestination.first << " and the state is "
         << travelDestination.second << endl;

    return 0;
}