
from jiwer import wer
import json
from rouge import Rouge


reference =["Corresponde a Castro y Roque demostrar de qué madera están hechos y subirse al carro.",
"A pesar de saber que era casi imposible ganar, Pedro seguía luchando por una causa perdida.",
"Después de varios intentos fallidos, Juan decidió tirar la toalla y dejar el curso de francés.",
"Perdí mi trabajo, pero conseguí una oportunidad aún mejor. Es verdad que cuando se cierra una puerta, se abre una ventana.",
"Mejor no busques problemas investigando eso, recuerda que la curiosidad mató al gato.",
"Consiguió ese trabajo no por talento, sino porque tiene enchufe en la empresa.",
"Deberías ser agradecido con tu jefe y no criticarlo tanto; no muerdas la mano que te da de comer.",
"Es curioso que el mecánico tenga su coche averiado. Como se dice, en casa del herrero, cuchillo de palo.",
"Decidí quedarme en mi trabajo actual; más vale lo malo conocido que lo bueno por conocer.",
"Los niños siempre se vuelven locos jugando cuando la maestra sale del aula. Cuando el gato no está, los ratones bailan.",
"Ella siempre tiene ideas fuera de lo común; en la escuela la consideraban un bicho raro.",
"El restaurante al que fuimos a cenar estaba en el quinto pino, nos llevó casi una hora llegar.",
"Después de vender su empresa, ahora está forrado y puede comprar lo que quiera.",
"El viejo coche de Carlos finalmente estiró la pata en medio del viaje.",
"Juan es tan bueno en matemáticas como lo fue su padre; de tal palo, tal astilla.",
"Al comprar los boletos de tren con antelación, ahorramos dinero y aseguramos nuestro viaje, matando dos pájaros de un tiro.",
"Prefiero no saber lo que pasó en esa fiesta porque, como dice el dicho, ojos que no ven, corazón que no siente.",
"Ese reloj parece muy caro, pero recuerda que no es oro todo lo que reluce.",
"Aunque en su pueblo era un don nadie, se mudó a la ciudad y logró hacerse famoso.",
"Después de hablar con mi amiga sobre mis problemas, me sentí mejor; es cierto que las penas compartidas saben a menos.",
"Juan decidió salir del armario y contarles a sus padres sobre su orientación sexual.",
"Cuando olvidó nuestro aniversario, fue la gota que colmó el vaso y decidí terminar la relación.",
"Ella adora la música clásica y él el rock pesado, pero sobre gustos no hay nada escrito.",
"Estaba como unas castañuelas cuando le dieron la noticia de que había pasado el examen.",
"Parecía un lugar sencillo y sin pretensiones, pero la comida era increíblemente buena. Sin duda, las apariencias engañan.",
"Me gusta probar comidas de diferentes culturas porque en la variedad está el gusto.",
"Se convirtió en un pez gordo de la industria después de que su invento revolucionara el mercado.",
"Cuando intentas abrir ese frasco, recuerda que más vale maña que fuerza.",
"Siempre me gusta hablar claro, al pan, pan y al vino, vino; así todos sabemos a qué atenernos.",
"Gracias por ayudarme con la mudanza. Recuerda, hoy por ti, mañana por mí.",
"Decidió invertir en un negocio nuevo, pensando que quien no arriesga, no gana.",
"No te preocupes por sus amenazas, ya sabes que perro ladrador, poco mordedor.",
"Siempre llevo un paraguas en el bolso, porque más vale prevenir que curar.",
"Los niños estaban haciendo el mono en el patio y no paraban de reírse.",
"Aunque me despidieron, encontré un trabajo mejor, así que no hay mal que por bien no venga.",
"No importa qué método uses para estudiar, todos los caminos llevan a Roma.",
"Es mejor no tratar con él cuando está enfadado porque tiene mala leche.",
"En la fiesta, Juan contó un chiste para romper el hielo y todos se relajaron.",
"Mira a esos políticos corruptos juntos; dios los cría y ellos se juntan.",
"Finalmente limpiaste tu cuarto, más vale tarde que nunca.",
"Ella no solo es mi esposa, también es mi media naranja.",
"Metí la pata cuando olvidé el aniversario de nuestra boda.",
"Cambió de equipo cuando los suyos empezaron a perder; es un verdadero chaquetero.",
"No necesito explicarte el por qué, a buen entendedor, pocas palabras bastan.",
"Salí un momento y al volver mi hermano había ocupado mi lugar en el sofá. Quien fue a Sevilla, perdió su silla.",
"En la celebración se pilló un pedo; no podía ni hablar.",
"Mi tío siempre anda haciendo cosas raras, está como una cabra.",
"No te preocupes por el examen de mañana, va a ser pan comido.",
"Voy a llevarme un paraguas por si las moscas, parece que podría llover.",
"Juan siempre está haciendo la pelota al jefe para conseguir lo que quiere.",
"Cuando se enfermó su colega, María arrimó el hombro para ayudar a terminar el proyecto.",
"En ese grupo grande, me siento como un cero a la izquierda, nadie nota si estoy o no.",
"Ese reloj me costó un ojo de la cara, pero vale cada centavo.",
"Mi hermano me llama cada dos por tres; es realmente pegajoso.",
"No compliques la situación buscándole tres pies al gato, es bastante simple.",
"En la reunión éramos cuatro gatos, casi nadie asistió.",
"El hotel estaba en el quinto pino, tuvimos que caminar muchísimo para llegar al centro.",
"Desde que comenzó el invierno, parece que me han entrado los siete males, no he parado de enfermarme.",
"Ese coche nuevo es más chulo que un ocho, todos se giran a mirarlo.",
"Se metió en camisas de once varas al tratar de mediar en esos problemas familiares.",
"Aunque todos le aconsejan lo contrario, él sigue en sus trece de no vender la casa.",
"Ese político es un chorizo, siempre está involucrado en escándalos de corrupción.",
"¡Tu móvil es del año de la pera!",
"Los soldados novatos a menudo son carne de cañón en los conflictos grandes.",
"No hables con el jefe hoy, está mala uva porque han bajado las ventas.",
"Después de tantos meses en el gimnasio, ahora está como un fideo.",
"Con toda esa lluvia, llegué a casa y estaba como una sopa.",
"A él le importa un pimiento lo que los demás piensen de su estilo.",
"¡Ostras! ¿Viste eso? ¡El coche casi se estrella!",
"Estoy hasta las narices de tus quejas.",
"Solo de pensar en la paella que hace mi abuela se me hace la boca agua.",
"Con todas estas deudas, estoy con el agua al cuello.",
"Si necesitas ayuda con la mudanza, yo te puedo echar una mano.",
"Se metió hasta la cabeza en el proyecto y no salía de su oficina en días.",
"Estaba tan nervioso durante la entrevista que no daba pie con bola.",
"Aunque recibió malas noticias, puso el mal tiempo, buena cara delante de sus hijos.",
"Ella puede hablar por los codos, nunca se queda sin temas de conversación.",
"Hoy me levanté con mal pie y todo me ha ido mal.",
"Estaba tan preocupado que no pegué ojo en toda la noche.",
"Cuando le dijeron que había ganado el premio, estaba loco con contento.",
"Desde que perdió el empleo, parece que no puede catch a break; ya sabes, perro flaco, todo son pulgas.",
"Terminamos nuestra relación amistosamente; fue bonito mientras duró.",
"No es el tipo de carro que yo me compraría, pero cada loco con su tema.",
"Pensaron que me habían vencido, pero gané el juicio. Quien ríe el último, ríe mejor.",
"Siempre llega tarde a sus citas, es un caracol haciendo todo.",
"El amor es cosa de dos, y si ustedes se gustan yo no los voy a detener.",
"No te preocupes por quedarte a cenar, donde comen dos, comen tres.",
"No te agobies trabajando toda la noche, no por mucho madrugar amanece más temprano.",
"¿Cancelar nuestras vacaciones? ¡De ninguna manera!",
"Cuando le pregunté por el dinero, se fue por las ramas y nunca dio una respuesta clara.",
"Me hubiera gustado ganar, pero perdí. Así es la vida.",
"No empieces a buscar problemas donde no los hay, no le busques pulgas al perro.",
"Decidimos dejar atrás los problemas y hacer borrón y cuenta nueva en nuestra amistad.",
"Después de la ruptura, fue al bar a ahogar las penas con alcohol.",
"Quiero que nuestra relación avance a la siguiente etapa, pero todo a su debido tiempo.",
"Después de trabajar doce horas seguidas, estaba frito.",
"En la fiesta, intentó ligarse a alguien con su nuevo look.",
"Si necesitas ayuda con la mudanza, cuenta conmigo.",
"Cuando sugirió cambiar la estrategia de marketing, realmente dio en el clavo.",
"Comenzó a llover justo cuando salimos, ya estamos otra vez.",
"El martes es festivo, así que vamos a hacer puente y no volveremos al trabajo hasta el miércoles.",
"Fui a comprar entradas, pero cerraron justo antes de que llegara y me dieron con la puerta en las narices.",
"A pesar de sus constantes fracasos, siempre se las arregla para volver: mala hierba nunca muere.",
"Después de trabajar todo el día sin comer, a buen hambre no hay pan duro; este pan viejo sabe delicioso.",
"Con esta tormenta, llegué a casa hecho una sopa.",
"Ha perdido tanto peso que está como un fideo.",
"Después de meses en el gimnasio, está como un queso.",
"No me importa un pepino.",
"Se puso como un tomate cuando olvidó las líneas en la obra.",
"¡Tu móvil es del año de la pera!",
"No te preocupes por ese examen; será pan comido para ti.",
"Dudo que estén despiertos aún. Se acuestan con las gallinas.",
"Desde que perdió el empleo, le han surgido problemas de todos lados; a perro flaco, todo son pulgas.",
"El vendedor de coches intentó darme gato por liebre.",
"Mi hijo no hace más que quejarse y rebelarse; definitivamente está en la edad del pavo.",
"Estás muy callado hoy, ¿se te ha comido la lengua el gato?",
"Esa computadora nueva me costó un ojo de la cara, pero es de última generación.",
"Estoy hasta las narices de tus quejas.",
"Mi tía habla hasta por los codos; puede pasar horas contando historias.",
"Estaba tan nerviosa por la entrevista que no pegué ojo en toda la noche.",
"Tu argumento no tiene ni pies ni cabeza, deberías organizar mejor tus ideas.",
"Ella no tiene pelos en la lengua, siempre dice lo que piensa, sin importar las consecuencias.",
"Me quemé las pestañas estudiando para el examen final de física.",
"Esos dos son uña y carne; siempre se ven juntos.",
"Mi amigo me dijo que conoció a Kim Kardashian, pero creo que me estaba tomando el pelo.",
"No te ahogues en un vaso de agua. El estrés es malo para la salud.",
"Mi novia me dejó por su ex. - Al mal tiempo, buena cara.",
"Estás complicando demasiado la situación, estás buscándole tres pies al gato.",
"Debes actuar rápido y decidido, recuerda que camarón que se duerme se lo lleva la corriente.",
"No sé qué decisión tomar respecto al nuevo contrato; voy a consultarlo con la almohada.",
"Es irónico que el mecánico tenga su coche siempre averiado, en casa del herrero, cuchillo de palo.",
"Perdí mi trabajo, pero luego me ofrecieron uno mejor, justo lo que dicen: cuando se cierra una puerta, se abre una ventana.",
"Puedes prometerme lo que sea, pero recuerda que las palabras se las lleva el viento.",
"Decidí aceptar la oferta de trabajo actual en lugar de esperar una mejor: más vale pájaro en mano que ciento volando.",
"Se metió en camisa de once varas al tratar de organizar una boda en un mes.",
"No te preocupes por esa ruptura amorosa; un clavo saca a otro clavo.",
"Él intentó darme consejos financieros, pero realmente debería seguir su propio campo—zapatero, a tus zapatos.",
"No me gusta hacer mucho ruido al salir, prefiero despedirme a la francesa.",
"El hijo de Roberto tuvo una aventura. - De tal palo, tal astilla.",
"Mira a esos tramposos jugando cartas juntos; Dios los cría y ellos se juntan.",
"Después de ganar el premio, se durmió en los laureles y dejó de esforzarse.",
"Hacer más tareas de las necesarias es como echar agua al mar.",
"Discutir sobre política solo echa leña al fuego en las reuniones familiares.",
"Debe faltarle un tornillo para pensar que puede correr un maratón sin entrenamiento.",
"Los políticos tienden a irse por las ramas cuando tienen que abordar temas polémicos.",
"Después de ver los beneficios, todos se quisieron subir al carro de la inversión tecnológica.",
"Ella siempre es optimista, ve todo color de rosa.",
"Estaba como unas castañuelas después de recibir la noticia de su promoción.",
"Está de mala leche hoy, mejor no le hables mucho.",
"Después de la maratón, estaba frito; no podía mover ni un músculo.",
"¿Quieres ir al gimnasio? - No. Estoy hecho polvo.",
"Cuando se enteró de la traición, estaba hecho un ají.",
"Estaba loco de contento cuando se enteró que iba a ser padre.",
"Evita hablar con el jefe hoy, está de mala uva por los resultados de las ventas.",
"Me quedé de piedra cuando anunciaron su compromiso.",
"No es el tipo de carro que yo me compraría, pero cada loco con su tema.",
"Cuando Jorge sugirió que el problema estaba en el software y no en el hardware, realmente dio en el clavo.",
"Estábamos en el picnic cuando empezó a llover a cántaros; tuvimos que empacar todo rápidamente y buscar refugio.",
"Metí la pata al decirle a Sara que la fiesta era sorpresa; no sabía que era un secreto.",
"Al estudiar español, estoy matando dos pájaros de un tiro: mejorando mi curriculum y preparándome para el viaje a México.",
"Perdió los estribos cuando vio que el perro había destrozado los cojines del sofá.",
"El papá de la cumpleañera tiró la casa por la ventana con esta fiesta.",
]




# file = "./data/json/spanish_baseline_prompt_output.json"
# file = "./data/json/spanish_step_by_step_prompt_output.json"
# file = "./data/json/spanish_cot_prompt_output.json"
    
with open(file,'r') as file:  
    data = json.load(file)
    
    
# # Initialize Rouge object
rouge = Rouge()
hypothesis = []
for obj in data:
    modified = obj["generation"].replace("\n", "")
    try: 
        modified = modified.split("Step 3. ")[1]
        modified = modified.split(" '")[1]
        modified = modified.replace(".'", ".")
        # modified = modified.replace("Spanish translation: ", "")
        # modified = modified.replace("Spanish translation of the sentence:","")
        hypothesis.append(modified)
    except:
        hypothesis.append("")
        
    
# # Calculate ROUGE-1 scores for each pair of hypothesis-reference summaries
total_scores =0.0
count_empty=0
for n in range(len(hypothesis)):
        try:
            scores = rouge.get_scores(hypothesis[n], reference[n], avg=True)
            total_scores += scores['rouge-1']['f']
            print(scores['rouge-1']['f'])
            print(f"ROUGE-1 F1 Score for pair {n}: {scores['rouge-1']['f']}")
        except:
            count_empty +=1

avg_rouge_combined =total_scores/float(len(hypothesis)-count_empty)
print(avg_rouge_combined)






count =0
total_error = 0

for i,item in enumerate(data):
    # modified_string = item["generation"].replace("\n", "")
    generation = item["generation"]
    step_3 = generation.split("\nStep 3.") # Selecting the part after "Step 3. "
    try:
        modified_string = step_3[1] # Extracting the sentence between single quotes
        # modified_string=modified_string.replace("Full sentence translates to '", "")
        # modified_string = modified_string=modified_string.replace(".'",".")
        modified_string = modified_string.replace("Spanish translation: ", "")
        modified_string = modified_string.replace("Spanish translation of the sentence:","")
        modified_string = modified_string.replace('"', "")
        # modified_string = modified_string.split(" '")[1]
        # modified_string = modified_string.replace(".'", ".")
        print(modified_string)
    except:
        modified_string =''
        count+=1
    if wer(reference[i], str(modified_string)) == 1.0:
        total_error += 0
    total_error += wer(reference[i], str(modified_string))
    
agg_wer=total_error/(len(data)-count)
print(agg_wer)

# wer baseline:0.41102913538342395
# wer sbs : 0.519475082039055
# wer cot: 0.42685347971800114

# avg rouge baseline: 0.6849774220549566
#avg sbs : 0.6797736588834958
# avg cot: 0.7010305853761392


