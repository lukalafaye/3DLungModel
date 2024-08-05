import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.spatial as ss
import random as rd

uX = np.array([1, 0, 0])
uY = np.array([0, 1, 0])
uZ = np.array([0, 0, 1])

proba = -1

global R0
global V0
global volume 

R0 = 1
V0 = 1
volume = 0

with open("x.txt", "a") as f:
    f.truncate(0)

with open("y.txt", "a") as f:
    f.truncate(0)

with open("z.txt", "a") as f:
    f.truncate(0)
    
with open("coordinates.csv", "a") as f:
    f.truncate(0)

##################################################################
### FONCTIONS
##################################################################

def creer_cylindreY(r, l, y0): 
    """
    Crée un cylindre d'axe de révolution (0,y), de rayon r et de longueur l dans le plan (O,x,y)  
    """
  
    M = 20 # Nombre de points sur un cercle
    N = 10 # Nombre de cercles sur un cylindre

    u = np.linspace(0, 2*np.pi, M) # Génère M angles uniformément espacés entre 0 et 2pi
    v = np.linspace(y0, y0+l, M) # Génère M valeurs de y uniformément espacées entre y0 et y0+l

    # On génère M coordonnées suivant (0,x) et (O,z) utilisées pour tracer un cercle d'ordonnée constante suivant (O,y)
    # Les coordonnées de même indice i dans les tableaux x, y et z permettent de tracer chacun des M points (xi, yi, zi) de ce cercle dans l'espace
    x = r * np.cos(u) 
    z = r * np.sin(u) 
    y = v 

    # On recopie les tableau x et z N fois chacun pour obtenir X et Z
    X = np.array(list(x)*N) # X = [x, x, x, ..., x] N fois
    Z = np.array(list(z)*N) # Z = [z, z, z, ..., z] N fois

    # On recopie chaque valeur de y N fois pour obtenir Y
    Y = np.repeat(y, N) # Y = [y0, ..., y0, y1, ..., y1, ..., yN, ..., yM] N fois 
    
    # Le nombre de lignes de X, Y et Z est (N x M) 
    # Les coordonnées de même indice i dans les tableaux X, Y et Z permettent de tracer chacun des NxM points (xj, yj, zj) du cylindre dans l'espace

    # On regroupe les coordonnées de même indice en empilant les tableaux X, Y et Z en tant que colonnes d'un nouveau tableau cyl
    cyl = np.column_stack((X, Y, Z)) # cyl = [  [x0, y0, zo], [x1, y0, z1], ..., [xM, y0, zM], ..., [x0, yN, z0], ..., [xM, yN, zM]   ] 

    return cyl

##################################################################

def créer_cylindre_rev(uRev, M, r, l):
    """
    Crée un cylindre d'axe de révolution (M, uRev) avec M = (x0, y0, z0) le centre d'une des bases du cylindre,  
    de longueur l et de rayon r, à l'aide de la fonction creer_cylindreY()

    On construit dans un premier temps le cylindre sur l'axe (O,y), ensuite on fait tourner son axe de révolution pour qu'il repose sur l'axe (M, uRev)
    Enfin, on translate le cylindre de OM, distance de M à l'origine
    """
    
    theta = angle(uY, uRev) # Angle entre l'axe de révolution voulu et (0,y)
    
    cyl = creer_cylindreY(r,l,0)
    cyl = rotation(cyl, uZ, theta)
    cyl = translation(cyl, M)
    
    return cyl

def créer_cylindres_filles(cyl1, uRev1, M, D, l, h, theta2, theta3, N, blocked, ax, trace):
    global volume
    global proba
    
    if N==1:
        return

    D_sis = h * D
    L_sis = l*1/2
    
    r2 = D_sis/2
    r3 = D_sis/2
    
    l2 = L_sis
    l3 = L_sis
    
    uRev2 = rotation(uRev1, uZ, -theta2)
    uRev3 = rotation(uRev1, uZ, theta3)
    
    # h2 = (r3+np.cos(theta2+theta3)*r2)/(np.sin(theta2+theta3))
    # h3 = (r2+np.cos(theta2+theta3)*r3)/(np.sin(theta2+theta3))
    
    h2 = r2 
    h3 = r2 

    #  yB = -h2*np.cos(theta2)+r2*np.sin(theta2)
    #  yD = -h3*np.cos(theta3)+r3*np.sin(theta3) #cf these mauroy fin
    #  h1 = max(yB,yD)

    cyl2 = créer_cylindre_rev(uRev2, M, r2, l2)
    cyl2 = translation(cyl2, uRev2*h2)
    
    cyl3 = créer_cylindre_rev(uRev3, M, r3, l3)
    cyl3 = translation(cyl3, uRev3*h3)
    
    r = rd.uniform(0,1)
    
    if blocked or r<=proba:
        blocked = 1

    else:
        blocked = 0
    
    color = "blue"
    
    if blocked:
        color = "black"
        trace = False
        plot(cyl2, r2, l2, ax, color, trace)
        plot(cyl3, r3, l3, ax, color, trace)
    else:
        volume += plot(cyl2, r2, l2, ax, color, trace)
        volume += plot(cyl3, r3, l3, ax, color, trace)
        #print(volume)
        
    M2 = translation(M, uRev2*l2)
    créer_cylindres_filles(cyl2, uRev2, M2, D_sis, l, h, theta2, theta3, N-1, blocked, ax, trace)
    
    M3 = translation(M, uRev3*l3)
    créer_cylindres_filles(cyl3, uRev3, M3, D_sis, l, h, theta2, theta3, N-1, blocked, ax, trace)

##################################################################

def créer_segment(A, e, N):
    """
    Crée un segment sur la droite portée par e = (ex,ey,ez) et passant par A, dont une des extrémités est A
    Un segment est un tableau discret de N points
    """
    
    t = normaliser(e)
    
    current_point = A # A = (x0, y0,z0)
    segment = np.array([current_point]) # Crée un tableau segment dans lequel on rajoutera les points

    for n in range(N): # La distance séparant deux points consécutifs est 1
        current_point = translation(current_point, t) # Effectue une translation du point de t pour obtenir le point suivant
        segment = np.vstack((segment, current_point)) # Rajoute le nouveau point à la fin du tableau segment
    return segment # Retourne le tableau contenant tous les points

##################################################################

def segment(A,B,N):
    """
    Crée un segment de N points d'extrémités A et B
    Un segment est un tableau discret de N points
    """
    e = normaliser(B-A)
    q=np.multiply(e,(norme(B-A)/N))

    current_point = A 
    segment = np.array([current_point]) # Stores current point in a new array

    for n in range(N): # La distance séparant deux points consécutifs est 1
        current_point = translation(current_point, q) # Effectue une translation du point de q pour obtenir le point suivant
        segment = np.vstack((segment, current_point)) # Rajoute le nouveau point à la fin du tableau segment
    return segment # Retourne le tableau contenant tous les points

##################################################################

def rotation(coords, e, theta):
    theta *= -1
    """
    Fait pivoter un objet 3D représenté par un tableau de coordonnées en utilisant le vecteur e = (ex, ey, ez) et l'angle theta
    e = (ex, ey, ez) doit être normealisé
    """
    
    u = normaliser(e)
    ux = u[0]
    uy = u[1]
    uz = u[2]
    
    
    P = np.array((
    (ux**2, ux*uy, ux*uz),
    (ux*uy, uy**2, uy*uz),
    (ux*uz, uy*uz, uz**2))
    )
    
    I = np.eye(3, dtype=float)
    
    Q = np.array((
    (0, -uz, uy),
    (uz, 0, -ux),
    (-uy, ux, 0))
    )
    
    R = P + np.cos(theta) * (I - P) + np.sin(theta) * Q 
    
    return np.dot(coords, R) # Renvoie un nouvel array représentant l'objet après rotation. (multiplie R et coord)

##################################################################

def translation(coord, t):
    """
    Translate un objet de t = [dx, dy, dz]
    """
    
    if coord.ndim == 1:
      return coord + t
    
    T = np.tile(t, (1, 1)) # Pour la dimension 2
    
    return T + coord

##################################################################

def angle(e, f):
    """
    Calcule l'angle entre les vecteurs e = (ex, ey, 0) et f = (fx, fy, 0)
    L'angle doit être entre -pi/2 et pi/2
    """
    
    u = normaliser(e)
    ux = u[0]
    uy = u[1]
    
    v = normaliser(f)
    vx = v[0]
    vy = v[1]
    
    prod_scal = np.dot([ux, uy, 0], [vx, vy, 0])
    
    theta = np.arccos(prod_scal)
    if (ux*vy-vx*uy) < 0: # Le produit vectoriel est négatif
        theta *= -1
    
    return theta

##################################################################

def norme(v):
    """
    Calcule la norme du vecteur v = (x,y,z)
    """
    
    x = v[0]
    y = v[1]
    z = v[2]
    
    return np.sqrt(x**2 + y**2 + z**2)

##################################################################

def normaliser(v):
  
  """
  Normalise le vecteur v = (x, y, z)
  """
  
  x = v[0]
  y = v[1]
  z = v[2]
  
  n = norme(v)
  
  return [x/n, y/n, z/n]

##################################################################

def plot(coords, r, l, ax, color, trace):
  x = coords[:,0] # Extrait les coordonnées x 
  y = coords[:,1] # coordonnées y
  z = coords[:,2] # coordonnées z

  with open("x.txt","a") as xf:
    np.savetxt(xf, x)

  with open("y.txt","a") as yf:
    np.savetxt(yf, y)

  with open("z.txt","a") as zf:
    np.savetxt(zf, z)
    
  if trace == True:
      ax.scatter(x, y, z, marker="o") # "c=color"

  combined_coords = np.column_stack((x, y, z))

  # Save combined coordinates to a CSV file
  with open("coordinates.csv", "a") as csv_file:
    np.savetxt(csv_file, combined_coords, delimiter=",")  # ss.ConvexHull(coords).volume

  return 0
  
############################################
### Construire l'arbre
############################################

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.get_zaxis().set_visible(False)

ax.set_xlim3d(-30, 30)
ax.set_ylim3d(-30, 30)
ax.set_zlim3d(-30, 30)

ax.azim = -45   # z rotation (default=270)
ax.elev = -60   # x rotation (default=0)
# ax.dist = 1    # define perspective (default=10)

############################################
# Obtention de hc=0.79
############################################
"""
# RES = 0
# VOL = 0

H = np.array([])
Rv = np.array([])
Rr = np.array([])
for h in np.linspace(0.7,0.9,50): 
   VOL = 0
   RES = 0
   for p in range(1,9):
       RES += (1/(2**p)) * (1/(h**(3*p)))
   RES += 1
   RES *= R0
   for p in range(1,9):
       VOL += (2**p) * h**(3*p)
   VOL += 1
   VOL *= V0
   H = np.append(H, h)
   Rv = np.append(Rv, np.array(VOL))
   Rr = np.append(Rr, np.array(RES))
ax1 = fig.add_subplot(111)
ax1.plot(H, Rv, '--r', label='Volume')
ax1.plot(H, Rr, '--b', label='Résistance')

ax1.set_xlabel('Facteur homothétique h')
ax1.set_ylabel('Résistance / Volume')

plt.title('Résistance et volume en fonction du facteur de réduction')

fig.savefig("plot.png")
"""
############################################
# Tracé 3D
############################################

uRev1 = uY
d = 4
l = 16
M = np.array([0, l, 0])
seg = segment(M, (0,0,0), 10)
plot(seg, d/2, l, ax, "blue", True)

theta = np.radians(45)
theta2=theta
theta3=theta

mother_cyl = creer_cylindreY(d/2, l, 0)

plot(mother_cyl, d/2, l, ax, "blue", True)
créer_cylindres_filles(mother_cyl,uRev1,M,d,l,0.79,theta2,theta3,4,0,ax,trace=True)

fig.savefig("plot.png")

print("Tracé")
