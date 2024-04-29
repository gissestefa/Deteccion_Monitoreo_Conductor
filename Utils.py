def convertirResultadoEntrenamiento(res):
    if (res==0):
        return "Abajo"
    elif(res==1):
        return "Arriba"
    elif(res==2):
        return "Correcta"
    elif(res==3):
        return "Derecha"
    elif(res==4):
        return "Izquierda"
    else:
        return "Incorrecto"