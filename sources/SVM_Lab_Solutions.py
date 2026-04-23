{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "header",
   "metadata": {},
   "source": [
    "# Support Vector Machines (SVM) Lab - SOLUTIONS\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e7ee20c6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "setup-header",
   "metadata": {},
   "source": [
    "# Part 1: Setup and Helper Functions\n",
    "\n",
    "Run the following cells to import necessary libraries and define helper functions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5cd9f72c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.datasets import make_circles, load_wine\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "from sklearn.inspection import DecisionBoundaryDisplay\n",
    "from sklearn.decomposition import PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec0fd648",
   "metadata": {},
   "outputs": [],
   "source": [
    "###############################################################################\n",
    "# Helper Function: Plot decision boundaries\n",
    "###############################################################################\n",
    "\n",
    "def plot_data_with_decision_boundary(model, X, y, scaler, ax, title=\"\"):\n",
    "    \"\"\"\n",
    "    Plots the decision boundary for a trained SVC in the original 2D coordinate system.\n",
    "    \n",
    "    model  : Already-fitted SVC model (trained on scaled data).\n",
    "    X, y   : The *original* (unscaled) features and labels for visualization.\n",
    "    scaler : The StandardScaler used to scale the data during training.\n",
    "    ax     : A matplotlib Axes object for plotting.\n",
    "    title  : Title for the plot.\n",
    "    \"\"\"\n",
    "    num_classes = len(np.unique(y))\n",
    "    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5\n",
    "    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5\n",
    "\n",
    "    ax.set_xlim(x_min, x_max)\n",
    "    ax.set_ylim(y_min, y_max)\n",
    "\n",
    "    xx, yy = np.meshgrid(\n",
    "        np.linspace(x_min, x_max, 200),\n",
    "        np.linspace(y_min, y_max, 200),\n",
    "    )\n",
    "    grid = np.c_[xx.ravel(), yy.ravel()]\n",
    "    grid_scaled = scaler.transform(grid)\n",
    "\n",
    "    DecisionBoundaryDisplay.from_estimator(\n",
    "        estimator=model,\n",
    "        X=grid_scaled,\n",
    "        response_method=\"predict\",\n",
    "        plot_method=\"pcolormesh\",\n",
    "        alpha=0.3,\n",
    "        ax=ax\n",
    "    )\n",
    "    \n",
    "    if num_classes == 2:\n",
    "        DecisionBoundaryDisplay.from_estimator(\n",
    "            estimator=model,\n",
    "            X=grid_scaled,\n",
    "            response_method=\"decision_function\",\n",
    "            plot_method=\"contour\",\n",
    "            levels=[-1, 0, 1],\n",
    "            colors=[\"k\", \"k\", \"k\"],\n",
    "            linestyles=[\"--\", \"-\", \"--\"],\n",
    "            ax=ax\n",
    "        )\n",
    "    else:\n",
    "        DecisionBoundaryDisplay.from_estimator(\n",
    "            estimator=model,\n",
    "            X=grid_scaled,\n",
    "            response_method=\"predict\",\n",
    "            plot_method=\"contour\",\n",
    "            levels=[-1, 0, 1],\n",
    "            colors=[\"k\", \"k\", \"k\"],\n",
    "            linestyles=[\"--\", \"-\", \"--\"],\n",
    "            ax=ax\n",
    "        )\n",
    "\n",
    "    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors=\"k\")\n",
    "    ax.legend(*scatter.legend_elements(), loc=\"upper right\", title=\"Classes\")\n",
    "    ax.set_title(title)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b5ae05e",
   "metadata": {},
   "source": [
    "# Part 2: SVM on Simulated Data (make_circles)\n",
    "\n",
    "In this section, we will:\n",
    "- Generate a synthetic dataset that is NOT linearly separable using `make_circles`\n",
    "- Compare Linear SVM vs. Polynomial SVM vs. RBF SVM\n",
    "- Observe how kernel choice impacts performance"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex3-1",
   "metadata": {},
   "source": [
    "## Exercise 2.1: Generate and Prepare the Data\n",
    "\n",
    "Complete the following tasks:\n",
    "1. Generate a synthetic dataset using `make_circles` with 300 samples, factor=0.5, noise=0.1, and random_state=42\n",
    "2. Split the data into training (70%) and testing (30%) sets with random_state=42\n",
    "3. Scale the features using `StandardScaler`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ee38999",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 3.1\n",
    "\n",
    "# Step 1: Generate a synthetic dataset (non-linear)\n",
    "X, y = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)\n",
    "\n",
    "# Step 2: Train-test split (70% train, 30% test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.3, random_state=42\n",
    ")\n",
    "\n",
    "# Step 3: Scale the data using StandardScaler\n",
    "scaler_sim = StandardScaler()\n",
    "X_train_scaled = scaler_sim.fit_transform(X_train)\n",
    "X_test_scaled = scaler_sim.transform(X_test)\n",
    "\n",
    "# Verify your work\n",
    "print(f\"Training set size: {X_train_scaled.shape[0]}\")\n",
    "print(f\"Test set size: {X_test_scaled.shape[0]}\")\n",
    "print(f\"Number of features: {X_train_scaled.shape[1]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex3-2",
   "metadata": {},
   "source": [
    "## Exercise 2.2: Train SVMs with Different Kernels\n",
    "\n",
    "Train three SVM models with different kernels:\n",
    "1. **Linear kernel**: `SVC(kernel=\"linear\")`\n",
    "2. **Polynomial kernel**: `SVC(kernel=\"poly\", degree=2)`\n",
    "3. **RBF kernel**: `SVC(kernel=\"rbf\")`\n",
    "\n",
    "Calculate and print the test accuracy for each model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "train-svms",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 3.2\n",
    "\n",
    "# Train Linear SVM\n",
    "svc_linear = SVC(kernel=\"linear\").fit(X_train_scaled, y_train)\n",
    "\n",
    "# Train Polynomial SVM (degree=2)\n",
    "svc_poly = SVC(kernel=\"poly\", degree=2).fit(X_train_scaled, y_train)\n",
    "\n",
    "# Train RBF SVM\n",
    "svc_rbf = SVC(kernel=\"rbf\").fit(X_train_scaled, y_train)\n",
    "\n",
    "# Calculate test accuracy for each model\n",
    "acc_lin = accuracy_score(y_test, svc_linear.predict(X_test_scaled))\n",
    "acc_poly = accuracy_score(y_test, svc_poly.predict(X_test_scaled))\n",
    "acc_rbf = accuracy_score(y_test, svc_rbf.predict(X_test_scaled))\n",
    "\n",
    "print(\"Simulated Data - Test Accuracy:\")\n",
    "print(f\"Linear: {acc_lin:.2f}\")\n",
    "print(f\"Poly:   {acc_poly:.2f}\")\n",
    "print(f\"RBF:    {acc_rbf:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex3-3",
   "metadata": {},
   "source": [
    "## Exercise 2.3: Visualize Decision Boundaries\n",
    "\n",
    "Use the provided `plot_data_with_decision_boundary` function to visualize the decision boundaries for all three kernels. Create a figure with 3 subplots (1 row, 3 columns)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "visualize-boundaries",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 3.3\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "models = [svc_linear, svc_poly, svc_rbf]\n",
    "titles = [\"Linear Kernel\", \"Polynomial Kernel\", \"RBF Kernel\"]\n",
    "\n",
    "for ax, model, t in zip(axes, models, titles):\n",
    "    plot_data_with_decision_boundary(model, X_test_scaled, y_test, scaler_sim, ax, t)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex3-4-header",
   "metadata": {},
   "source": [
    "## Exercise 2.4: Analysis Questions\n",
    "\n",
    "Based on your results above, answer the following questions:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex3-4-answers",
   "metadata": {},
   "source": [
    "**SOLUTION - Answers:**\n",
    "\n",
    "1. **Which kernel performed best?** The Polynomial and RBF kernels performed best (around 100% accuracy) because the data has a circular/radial structure that cannot be separated by a straight line. These kernels can capture the non-linear boundary.\n",
    "\n",
    "2. **Why did the linear kernel perform poorly?** The linear kernel can only create a straight-line decision boundary. The circles dataset has one class forming a ring around the other, making it impossible to separate with a linear hyperplane.\n",
    "\n",
    "3. **What do the dashed lines represent?** The dashed lines represent the margin boundaries (at distance ±1 from the decision hyperplane in the kernel space). Points on or within these margins are the support vectors that define the classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c697a92f",
   "metadata": {},
   "source": [
    "# Part 3: Hyperparameter Tuning\n",
    "\n",
    "We saw that an RBF SVM generally performs better than a purely linear one on non-linear datasets. But how do we pick the best C or gamma?\n",
    "\n",
    "We'll use GridSearchCV to find the optimal hyperparameters."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex4-1",
   "metadata": {},
   "source": [
    "## Exercise 3.1: Perform Grid Search\n",
    "\n",
    "Complete the following:\n",
    "1. Define a parameter grid with:\n",
    "   - `C`: [0.1, 1, 10, 100]\n",
    "   - `gamma`: [0.001, 0.01, 0.1, 1]\n",
    "   - `kernel`: ['rbf']\n",
    "2. Create a GridSearchCV object with 5-fold cross-validation\n",
    "3. Fit the grid search on the training data\n",
    "4. Print the best parameters and evaluate on the test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bef8a2e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 4.1\n",
    "\n",
    "# Step 1: Define the parameter grid\n",
    "param_grid = {\n",
    "    'C': [0.1, 1, 10, 100],\n",
    "    'gamma': [0.001, 0.01, 0.1, 1],\n",
    "    'kernel': ['rbf']\n",
    "}\n",
    "\n",
    "# Step 2: Create GridSearchCV object\n",
    "grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)\n",
    "\n",
    "# Step 3: Fit on training data\n",
    "grid_search.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Step 4: Print results\n",
    "print(\"Best parameters:\", grid_search.best_params_)\n",
    "print(\"Best CV accuracy:\", grid_search.best_score_)\n",
    "\n",
    "# Evaluate on test set\n",
    "best_model = grid_search.best_estimator_\n",
    "test_acc = best_model.score(X_test_scaled, y_test)\n",
    "print(\"Test accuracy with best parameters:\", test_acc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c53eaf6b",
   "metadata": {},
   "source": [
    "# Part 4: Applying SVM to the Wine Dataset\n",
    "\n",
    "Now let's apply what we've learned to a real dataset. The Wine dataset contains the chemical analysis of wines grown in Italy, with three different types of wine (classes). This is a multi-class classification problem."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex5-1",
   "metadata": {},
   "source": [
    "## Exercise 4.1: Load and Explore the Data\n",
    "\n",
    "Load the Wine dataset and print:\n",
    "- Dataset shape\n",
    "- Number of classes\n",
    "- Class distribution\n",
    "- Feature names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96be28c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 4.1\n",
    "\n",
    "wine_data = load_wine()\n",
    "X_wine = wine_data.data\n",
    "y_wine = wine_data.target\n",
    "\n",
    "print(\"Wine dataset shape:\", X_wine.shape)\n",
    "print(\"Number of classes:\", len(np.unique(y_wine)))\n",
    "print(\"Class distribution:\", np.bincount(y_wine))\n",
    "print(\"Feature names:\", wine_data.feature_names)\n",
    "print(\"Classes:\", wine_data.target_names)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex5-2",
   "metadata": {},
   "source": [
    "## Exercise 4.2: Prepare the Data\n",
    "\n",
    "Split the Wine data into training (70%) and testing (30%) sets:\n",
    "- Use `random_state=42`\n",
    "- Use `stratify=y_wine` to ensure proportionate class splits\n",
    "- Scale the features using StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "502ddd19",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 4.2\n",
    "\n",
    "X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(\n",
    "    X_wine, y_wine,\n",
    "    test_size=0.3,\n",
    "    random_state=42,\n",
    "    stratify=y_wine  # ensures proportionate class splits\n",
    ")\n",
    "\n",
    "# Scale the data\n",
    "scaler_wine = StandardScaler()\n",
    "X_train_wine_scaled = scaler_wine.fit_transform(X_train_wine)\n",
    "X_test_wine_scaled = scaler_wine.transform(X_test_wine)\n",
    "\n",
    "print(f\"Training set size: {X_train_wine_scaled.shape[0]}\")\n",
    "print(f\"Test set size: {X_test_wine_scaled.shape[0]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex5-3",
   "metadata": {},
   "source": [
    "## Exercise 4.3: Tune Hyperparameters for Wine Classification\n",
    "\n",
    "Perform a grid search with the following parameter grid:\n",
    "- `C`: [0.01, 0.1, 1, 10, 100]\n",
    "- `gamma`: [0.001, 0.01, 0.1, 1, 10]\n",
    "- `kernel`: ['rbf']\n",
    "\n",
    "Use 5-fold cross-validation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wine-gridsearch",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 4.3\n",
    "\n",
    "param_grid_wine = {\n",
    "    'C': [0.01, 0.1, 1, 10, 100],\n",
    "    'gamma': [0.001, 0.01, 0.1, 1, 10],\n",
    "    'kernel': ['rbf']\n",
    "}\n",
    "\n",
    "grid_search_wine = GridSearchCV(SVC(), param_grid_wine, cv=5, scoring='accuracy', verbose=1)\n",
    "\n",
    "# Fit the grid search\n",
    "grid_search_wine.fit(X_train_wine_scaled, y_train_wine)\n",
    "\n",
    "print(\"Best parameters (Wine):\", grid_search_wine.best_params_)\n",
    "print(\"Best CV Accuracy (Wine):\", grid_search_wine.best_score_)\n",
    "\n",
    "best_svm_wine = grid_search_wine.best_estimator_\n",
    "test_acc_wine = best_svm_wine.score(X_test_wine_scaled, y_test_wine)\n",
    "print(\"Test Accuracy with best parameters (Wine):\", test_acc_wine)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex5-4",
   "metadata": {},
   "source": [
    "## Exercise 4.4: Evaluate the Model\n",
    "\n",
    "Using the best model from grid search:\n",
    "1. Generate predictions on the test set\n",
    "2. Print the confusion matrix\n",
    "3. Print the classification report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wine-evaluation",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 4.4\n",
    "\n",
    "y_pred_wine = best_svm_wine.predict(X_test_wine_scaled)\n",
    "\n",
    "# Print confusion matrix\n",
    "cm = confusion_matrix(y_test_wine, y_pred_wine)\n",
    "print(\"Confusion Matrix:\\n\", cm)\n",
    "\n",
    "# Print classification report\n",
    "cr = classification_report(y_test_wine, y_pred_wine, target_names=wine_data.target_names)\n",
    "print(\"\\nClassification Report:\\n\", cr)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "part6-header",
   "metadata": {},
   "source": [
    "# Part 5: Visualizing Decision Boundaries on Wine Data\n",
    "\n",
    "The Wine dataset has 13 features, which makes it impossible to visualize directly. We'll use PCA to reduce it to 2 dimensions for visualization purposes."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex6-1",
   "metadata": {},
   "source": [
    "## Exercise 5.1: Apply PCA for Visualization\n",
    "\n",
    "1. Apply PCA to reduce the scaled Wine data to 2 components\n",
    "2. Scale the PCA-transformed data (for better SVM performance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wine-pca",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 5.1\n",
    "\n",
    "pca = PCA(n_components=2)\n",
    "X_train_2d = pca.fit_transform(X_train_wine_scaled)\n",
    "X_test_2d = pca.transform(X_test_wine_scaled)\n",
    "\n",
    "# Scale the PCA-transformed data\n",
    "scaler_wine_PCA = StandardScaler()\n",
    "X_train_2d = scaler_wine_PCA.fit_transform(X_train_2d)\n",
    "X_test_2d = scaler_wine_PCA.transform(X_test_2d)\n",
    "\n",
    "print(f\"PCA-transformed training data shape: {X_train_2d.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex6-2",
   "metadata": {},
   "source": [
    "## Exercise 5.2: Train SVMs on 2D Data and Compare Kernels\n",
    "\n",
    "Train three SVM models on the 2D PCA-transformed data:\n",
    "1. Linear SVM with C=1.0\n",
    "2. Polynomial SVM with degree=3 and C=200.0\n",
    "3. RBF SVM with C=1.0\n",
    "\n",
    "Print the training and test accuracies for each."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wine-svm-2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 5.2\n",
    "\n",
    "svc_linear = SVC(kernel='linear', C=1.0).fit(X_train_2d, y_train_wine)\n",
    "svc_poly = SVC(kernel='poly', degree=3, C=200.0).fit(X_train_2d, y_train_wine)\n",
    "svc_rbf = SVC(kernel='rbf', C=1.0).fit(X_train_2d, y_train_wine)\n",
    "\n",
    "# Print accuracies\n",
    "print(\"Linear SVM (2D PCA) Train Accuracy:\", svc_linear.score(X_train_2d, y_train_wine))\n",
    "print(\"Linear SVM (2D PCA) Test Accuracy:\", svc_linear.score(X_test_2d, y_test_wine))\n",
    "\n",
    "print(\"Polynomial SVM (2D PCA) Train Accuracy:\", svc_poly.score(X_train_2d, y_train_wine))\n",
    "print(\"Polynomial SVM (2D PCA) Test Accuracy:\", svc_poly.score(X_test_2d, y_test_wine))\n",
    "\n",
    "print(\"RBF SVM (2D PCA) Train Accuracy:\", svc_rbf.score(X_train_2d, y_train_wine))\n",
    "print(\"RBF SVM (2D PCA) Test Accuracy:\", svc_rbf.score(X_test_2d, y_test_wine))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ex6-3",
   "metadata": {},
   "source": [
    "## Exercise 5.3: Visualize All Three Kernels\n",
    "\n",
    "Create a figure with 3 subplots showing the decision boundaries for Linear, Polynomial, and RBF kernels on the Wine data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "wine-visualization",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Exercise 5.3\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "plot_data_with_decision_boundary(\n",
    "    model=svc_linear,\n",
    "    X=X_train_2d,\n",
    "    y=y_train_wine,\n",
    "    scaler=scaler_wine_PCA,\n",
    "    ax=axes[0],\n",
    "    title=r\"Wine Data (2D PCA) - Linear SVM\"\n",
    ")\n",
    "\n",
    "plot_data_with_decision_boundary(\n",
    "    model=svc_poly,\n",
    "    X=X_train_2d,\n",
    "    y=y_train_wine,\n",
    "    scaler=scaler_wine_PCA,\n",
    "    ax=axes[1],\n",
    "    title=r\"Wine Data (2D PCA) - Polynomial SVM (degree=3)\"\n",
    ")\n",
    "\n",
    "plot_data_with_decision_boundary(\n",
    "    model=svc_rbf,\n",
    "    X=X_train_2d,\n",
    "    y=y_train_wine,\n",
    "    scaler=scaler_wine_PCA,\n",
    "    ax=axes[2],\n",
    "    title=r\"Wine Data (2D PCA) - RBF SVM\"\n",
    ")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c5bfe9b",
   "metadata": {},
   "source": [
    "# Part 6: Observations & Conclusions\n",
    "\n",
    "Answer the following questions based on your experiments:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "final-questions",
   "metadata": {},
   "source": [
    "**SOLUTION - Answers:**\n",
    "\n",
    "1. **Kernel performance comparison:** On the simulated circles data, the linear kernel failed completely (~42% accuracy) while RBF and polynomial achieved ~100%. On the Wine data, all kernels performed reasonably well because the data is more linearly separable in high dimensions. The RBF kernel still edges ahead slightly due to its flexibility.\n",
    "\n",
    "2. **Optimal hyperparameters:** For the circles data, moderate values of C and gamma worked well. For the Wine data, typical best parameters are C=1 and gamma=0.1. Lower C values indicate the data is fairly separable (don't need to penalize misclassifications heavily). The moderate gamma suggests a balanced influence range.\n",
    "\n",
    "3. **2D vs 13D accuracy difference:** The accuracy on 2D PCA data is typically lower because PCA discards information when reducing from 13 to 2 dimensions. Some discriminative features may be lost in the projection. The full-dimensional SVM can leverage all chemical properties for classification.\n",
    "\n",
    "4. **Decision boundary complexity:** The polynomial kernel (especially with higher degrees) and RBF can create the most complex boundaries. However, more complexity is NOT always better - overly complex boundaries can overfit to training noise. The best model balances fitting the data well with generalizing to unseen examples (bias-variance tradeoff)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bonus-header",
   "metadata": {},
   "source": [
    "# Additional Exercises (Optional)\n",
    "\n",
    "Try the following extensions:\n",
    "1. Try polynomial kernels with different degrees on the Wine dataset\n",
    "2. Explore different cross-validation strategies (e.g., StratifiedKFold with different numbers of folds)\n",
    "3. Plot a heatmap to visualize how (C, gamma) pairs affect accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bonus",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SOLUTION: Heatmap of C-gamma pairs\n",
    "\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "\n",
    "# Extract results from grid search\n",
    "results = pd.DataFrame(grid_search_wine.cv_results_)\n",
    "\n",
    "# Create pivot table for heatmap\n",
    "pivot_table = results.pivot_table(\n",
    "    values='mean_test_score',\n",
    "    index='param_gamma',\n",
    "    columns='param_C'\n",
    ")\n",
    "\n",
    "# Plot heatmap\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')\n",
    "plt.title('Grid Search Accuracy Heatmap (C vs Gamma)')\n",
    "plt.xlabel('C')\n",
    "plt.ylabel('Gamma')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
