import requests as rq
import _thread


def download_json(rInicio, rFinal, no_thread):
    for i in range(rInicio,rFinal):
        for intentos in range(1,10):
            try:
                print(f"Thread {no_thread}: obteniendo", i)
                data = rq.get("https://resultados2019.tse.org.gt/201901/api.php?mesa=" + str(i) + "&vista=MESA&token=06124b3c608f20e29c7181c54f72df8fe62dadf0")
                if data.text[0] == "{":
                    with open("mesas_rv/"+str(i)+".json", "w") as ofile:
                        ofile.write(data.text)
                        print(f"Thread {no_thread}: guardado", i)
                        break
                else:
                    print(f"Thread {no_thread}: no se pudo obtener mesa (bad response) "+str(i))
            except Exception as e:
                print(f"Thread {no_thread}: no se pudo obtener mesa (exception) "+str(e))


no_threads = int(input("Ingrese numero de threads: "))

try:
    for x in range(no_threads):
        rInicio = int(input(f"Ingrese rango inicial para thread {x}: "))
        rFinal = int(input(f"Ingrese rango final para thread {x}: "))
        _thread.start_new_thread( download_json, (rInicio, rFinal, x, ) )
except Exception as e:
   print ("Error: unable to start thread")
   print (e)

while 1:
   pass
