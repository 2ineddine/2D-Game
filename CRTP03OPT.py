import numpy as np
from scipy.optimize import root,minimize
import matplotlib.pyplot as plt
import random

# Définir la fonction objectif
def j1(variable):
    x1,x2=variable
    return x1**2 + (1.5 * x2**2) - (3 * np.sin(2 * x1 + x2)) + (5 * np.sin(x1 - x2))


def gradient(variables):
    x1, x2 = variables
    grad_x1 = (2 * x1) - (6 * np.cos(2 * x1 + x2)) + (5 * np.cos(x1 - x2))
    grad_x2 = (3 * x2) - (3 * np.cos(2 * x1 + x2)) - (5 * np.cos(x1 - x2))
    return np.array([grad_x1, grad_x2])




def hessienne (racines):
    x1,x2=racines
    hessien=[[-5*np.sin(x1 - x2) + 12*np.sin(2*x1 + x2) + 2,        5*np.sin(x1 - x2) + 6*np.sin(2*x1 + x2)],
               [      5*np.sin(x1 - x2) + 6*np.sin(2*x1 + x2), -5*np.sin(x1 - x2) + 3*np.sin(2*x1 + x2) + 3.0]]
    return np.array(hessien)

def affiche_isovaleur_simple (f):
    # Créer des vecteurs x1 et x2
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    
    # Générer la grille de coordonnées
    X1, X2 = np.meshgrid(x1, x2)
    variables = [X1,X2]
    
    # Calculer les valeurs de j1 pour chaque point de la grille
    Z = f(variables)
    
    # Tracer les courbes de niveau (isovaleurs)
    plt.contour(X1, X2, Z, levels=65, cmap='viridis')
    
    # Ajouter des titres et labels
    plt.title("Courbes de niveau de la fonction j1(x1, x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    # Ajouter une barre de couleurs
    plt.colorbar(label="Valeur de j1(x1, x2)")
    
    # Afficher le graphique
    plt.show()



racine_gradient = []
val_propre = []
inter = 0

for i in range(-3, 3, 1):
    for j in range(-2, 2, 1):
        sol = root(gradient, [i, j])  # processus de la recherche du point critique par la fonction root
        if sol.success:
            # Vérification si sol.x (arrondi à 1 décimale) n'est pas déjà dans racine_gradient
            if not any(np.array_equal(np.round(sol.x, decimals=1), np.round(rg, decimals=1)) for rg in racine_gradient):
                # ajouter les dans un tableau de points critiques 
                racine_gradient.append(sol.x)

                # calcul des valeurs propres
                val_propre.append(np.linalg.eigvals(hessienne(sol.x)))

 



def affiche_isovaleur (f,racine_gradient):
    # Créer des vecteurs x1 et x2
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    
    # Générer la grille de coordonnées
    X1, X2 = np.meshgrid(x1, x2)
    variables = [X1,X2]
    
    # Calculer les valeurs de j1 pour chaque point de la grille
    Z = f(variables)
    
    # Tracer les courbes de niveau (isovaleurs)
    plt.contour(X1, X2, Z, levels=65, cmap='viridis')
    
    # Ajouter des titres et labels
    plt.title("Courbes de niveau de la fonction j1(x1, x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    for i in racine_gradient:
        plt.scatter(i[0],i[1],color='red', label='Points critiques', zorder=5)
    # Ajouter une barre de couleurs
    plt.colorbar(label="Valeur de j1(x1, x2)")
    
    # Afficher le graphique
    plt.show()
#affiche_isovaleur(j1,racine_gradient)
#nature des valeurs propres



initial_guesses = [
    [0, 0],  # Premier point de départ
    [1, 1],  # Deuxième point de départ
    [-1, -1],  # Troisième point de départ
    [2, 2],  # Quatrième point de départ
    [-2, -2],  # Cinquième point de départ
    [-5, -5],  # Cinquième point de départ
    [-4, -4]  # Cinquième point de départ
]
minimum_points_crts = []
for j in range (0,10):
    x0  = np.random.randint(-3,3,size = 2)
    resultat = minimize(j1,x0)
    if resultat.success:
        if not any(np.array_equal(np.round(resultat.x, decimals=1), np.round(rg, decimals=1)) for rg in minimum_points_crts):
            minimum_points_crts.append(resultat.x)
        #print(resultat)
#print(minimum_points_crts)

# partie 2 
#1
def methode_gradient_fixe (function,gradient,x0,alpha,epsilon,n_max):
    xn = np.array(x0)
    Xn_suite = [xn]
    n = 0
    dx = float('inf')
    convergence = False 
    
    while dx > epsilon and n < n_max:
        x_n1 = xn - alpha * gradient(xn)
        dx = np.linalg.norm(x_n1 - xn)
        xn = x_n1
        Xn_suite.append(xn)
        n += 1
    
    if dx <= epsilon:
        convergence = True
        
    return Xn_suite,convergence,n


    
def affichage_gradient_evolution (f,Xn_suite):

    # Créer des vecteurs x1 et x2
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    
    # Générer la grille de coordonnées
    X1, X2 = np.meshgrid(x1, x2)
    variables = [X1,X2]
    
    # Calculer les valeurs de j1 pour chaque point de la grille
    Z = f(variables)
    
    # Tracer les courbes de niveau (isovaleurs)
    plt.contour(X1, X2, Z, levels=65, cmap='viridis')
    
    # Ajouter des titres et labels
    plt.title("Courbes de niveau de la fonction j1(x1, x2)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    # Ajouter une barre de couleurs
    plt.colorbar(label="Valeur de j1(x1, x2)")

    # Tracer la suite des points Xn_suite
    for i in range(len(Xn_suite) - 1):
        plt.plot([Xn_suite[i][0], Xn_suite[i + 1][0]], 
                 [Xn_suite[i][1], Xn_suite[i + 1][1]], color='red', marker='+')
    plt.scatter(*zip(*Xn_suite), color='red', s=1)
    # Ajouter les points critiques sur le graphique
    #for point in Xn_suite:
        #plt.scatter(point[0], point[1], color='red')
       
    plt.show()












def methode_gradient_flexible (function,gradient,x0,alpha,epsilon,n_max):
    xn = np.array(x0)
    Xn_suite = [xn]
    n = 0
    dx = float('inf')
    convergence = False 
    
    while dx > epsilon and n < n_max:
        x_n1 = xn - alpha * gradient(xn)
        if function(x_n1)> function(xn):
            alpha*=0.5
        dx = np.linalg.norm(x_n1 - xn)
        xn = x_n1
        Xn_suite.append(xn)
        n += 1
    
    if dx < epsilon:
        convergence = True
        
    return Xn_suite,convergence,n,xn   

#Xn_suite,convergence1,n,xn = methode_gradient_flexible(j1, gradient, x0, alpha, epsilon, n_max)
#if convergence1:
#    affichage_gradient_evolution(j1, Xn_suite)
#else: 
#    print("la méthode gradient flexible n'a pas convergé ")


    

def NewtonMin(f, x0, epsilon, n_max):
    Xn = np.array(x0)          
    dX = 1                     
    n = 0                      
    converge = False           
    Xn_suite = [x0]           

    while dX > epsilon and n < n_max:  
        grad = gradient(Xn)           
        hess_inv = np.linalg.inv(hessienne(Xn))  
        delta_X = -np.dot(hess_inv, grad)  
        Xn = Xn + delta_X              
        Xn_suite.append(Xn.copy())     

        
        if n != 0:
            dX = np.linalg.norm(Xn - Xn_suite[-2])  
        else: 
            dX = np.linalg.norm(Xn - x0)  

        n += 1  

    if dX <= epsilon:
        converge = True  

    return converge, Xn_suite, n  





def afficher_isoval_pente(f, A, B):
    # Générer une plage de valeurs pour x1 et x2
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)

    # Calculer la pente et l'ordonnée de la droite passant par A et B
    slope = (A[1] - B[1]) / (A[0] - B[0])
    intercept = A[1] - slope * A[0]
    
    # Calculer les valeurs de y (x2) pour tracer la droite y = mx + b
    y = slope * x1 + intercept

    # Générer la grille de coordonnées
    X1, X2 = np.meshgrid(x1, x2)
    variables = [X1, X2]
    
    # Calculer les valeurs de la fonction f pour chaque point de la grille
    Z = f(variables)
    
    # Tracer les courbes de niveau (isovaleurs)
    contour = plt.contour(X1, X2, Z, levels=65, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=7, fmt='%1.1f')

    
    # Tracer la droite qui passe par A et B
    plt.plot(x1, y, color='red', linewidth=2, label="Droite passant par A et B")
    
    # Ajouter des titres et labels
    plt.title("Courbes de niveau et droite passant par A et B")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # Limiter les axes
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    # Afficher la légende
    plt.legend()

    # Afficher le graphique
    plt.show()
    
#afficher_isoval_pente (j1,[1,2],[3,2])  
    


def EgalityConstraint1(x1,x2):
    return x2 + (3/2) * x1
    
def EgalityConstraint2(x1,x2):
    return  x2 + (3/2) * x1 - 1


def LagrageEquation1(var):
    x1,x2,lambd  = var
    eq1 = 2 * x1 - (6 * np.cos(2 * x1 + x2 ))+ 5 * np.cos(x1 - x2) + (3/2) * lambd
    eq2 = 3 * x2 - 3 * np.cos(2 * x1 + x2) - 5 * np.cos(x1 - x2) + lambd
    eq3 = EgalityConstraint1(x1, x2)
    return [eq1,eq2,eq3]



def LagrageEquation2(var):
    x1, x2, lam = var
    eq1 = 2*x1 - 6*np.cos(2*x1 + x2) + 5*np.cos(x1 - x2) + (3/2)*lam
    eq2 = 3*x2 - 3*np.cos(2*x1 + x2) - 5*np.cos(x1 - x2) + lam
    eq3 = EgalityConstraint2(x1, x2)    
    return [eq1,eq2,eq3]

InitialPoint  = [1,1,1]

#solution = root(LagrageEquation,InitialPoint)
#Sol_x1,Sol_x2,Sol_lambd = solution.x
#print(solution.x)

A= [0,0]
B= [2,-3]
#DisplayIsvaluesSlab(j1, A, B)





def afficher_isoval_pente_solution(f, A, B,solution):
    # Générer une plage de valeurs pour x1 et x2
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    
    # Calculer la pente et l'ordonnée de la droite passant par A et B
    slope = (A[1] - B[1]) / (A[0] - B[0])
    intercept = A[1] - slope * A[0]
    
    # Calculer les valeurs de y (x2) pour tracer la droite y = mx + b
    y = slope * x1 + intercept

    # Générer la grille de coordonnées
    X1, X2 = np.meshgrid(x1, x2)
    variables = [X1, X2]
    
    # Calculer les valeurs de la fonction f pour chaque point de la grille
    Z = f(variables)
    
    # Tracer les courbes de niveau (isovaleurs)
    contour = plt.contour(X1, X2, Z, levels=65, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=7, fmt='%1.1f')
    # Tracer la droite qui passe par A et B
    plt.plot(x1, y, color='red', linewidth=2, label="Droite passant par A et B")
    plt.scatter(solution[0],solution[1], color= 'red',marker = 'o')
    # Ajouter des titres et labels
    plt.title("Courbes de niveau et droite passant par A et B")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # Limiter les axes
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    # Afficher la légende
    plt.legend()
    plt.show()

    # Afficher le graphiq
    
#InitialPoint = [1,1,1]
#C= [0,1]
#D= [2,-2]
#solution = root(LagrageEquation2,InitialPoint)
#Sol_x1,Sol_x2,Sol_lambd = solution.x #recupération de la solution de X1, X2 et lambda
#print(f"[{Sol_x1},{Sol_x2}] \n lambda = {Sol_lambd}")

#afficher_isoval_pente_solution (j1, C, D, [Sol_x1,Sol_x2])