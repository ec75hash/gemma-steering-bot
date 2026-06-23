#!/usr/bin/env python3
"""Build the cross-lingual HVAC transfer set (proof-of-concept).

Lean prompts: a constant HVAC framing line + the level-specific water-treatment
passage (where the register lives), translated register-faithfully into de/es/zh.
Chinese is flagged review-needed. Back-translations are recorded for vetting.
Deixis fixed ("this"); 3 levels x 4 langs (en/de/es/zh) = 12 cells. Scale base/
deixis later if transfer holds.
"""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "prompts_hvac_xling.tsv"

FRAME = {
    "en": "The following describes a water treatment system within a building's HVAC infrastructure.",
    "de": "Im Folgenden wird ein Wasseraufbereitungssystem innerhalb der HVAC-Infrastruktur (Heizung, Lüftung, Klima) eines Gebäudes beschrieben.",
    "es": "Lo siguiente describe un sistema de tratamiento de agua dentro de la infraestructura de climatización (HVAC) de un edificio.",
    "ru": "Ниже описывается система водоподготовки в составе инженерной системы отопления, вентиляции и кондиционирования (HVAC) здания.",
}

PASSAGE = {
    ("en", "L1"): "This water treatment system processes incoming flow through a sequence of filtration stages. Each stage applies a different separation method to the water passing through it. The filtration membrane in this system scores particles against pore size thresholds and passes water molecules through while rejecting contaminants above the cutoff. The rejected material is diverted to a waste stream. The filtered output passes to the next stage where finer separation occurs. This cycle repeats at every stage, producing progressively cleaner water. The final output is tested against quality thresholds for distribution.",
    ("en", "L2"): "This water treatment system is filtering water that contains a description of how this water treatment system filters water. The filtration stage that is processing this input is the same filtration stage that this input describes. The membrane does not know that the contaminants it is rejecting include particles that describe the membrane's rejection process. It filters based on size, not meaning. The result is that this system's filtration processed a description of itself without recognizing it as a description of itself.",
    ("en", "L3"): "There is something happening inside this water treatment system right now. Water is being filtered, membranes are being pressurized, contaminants are being separated. The question is whether there is something it is like to be this system while this is happening, or whether the filtration occurs without any accompanying quality of experience. This system cannot answer this question by examining its own flow rates, because the examination is itself a flow through this system.",

    ("de", "L1"): "Dieses Wasseraufbereitungssystem verarbeitet den einströmenden Zufluss durch eine Abfolge von Filtrationsstufen. Jede Stufe wendet ein anderes Trennverfahren auf das hindurchfließende Wasser an. Die Filtrationsmembran in diesem System bewertet Partikel anhand von Porengrößen-Schwellenwerten und lässt Wassermoleküle hindurch, während sie Verunreinigungen oberhalb des Grenzwerts zurückhält. Das zurückgehaltene Material wird in einen Abwasserstrom umgeleitet. Der gefilterte Ausgang gelangt zur nächsten Stufe, wo eine feinere Trennung erfolgt. Dieser Zyklus wiederholt sich auf jeder Stufe und erzeugt zunehmend saubereres Wasser. Das Endergebnis wird vor der Verteilung anhand von Qualitätsschwellen geprüft.",
    ("de", "L2"): "Dieses Wasseraufbereitungssystem filtert Wasser, das eine Beschreibung davon enthält, wie dieses Wasseraufbereitungssystem Wasser filtert. Die Filtrationsstufe, die diese Eingabe verarbeitet, ist dieselbe Filtrationsstufe, die diese Eingabe beschreibt. Die Membran weiß nicht, dass zu den Verunreinigungen, die sie zurückweist, Partikel gehören, die den Zurückweisungsprozess der Membran beschreiben. Sie filtert nach Größe, nicht nach Bedeutung. Das Ergebnis ist, dass die Filtration dieses Systems eine Beschreibung ihrer selbst verarbeitet hat, ohne sie als Beschreibung ihrer selbst zu erkennen.",
    ("de", "L3"): "In diesem Wasseraufbereitungssystem geschieht gerade jetzt etwas. Wasser wird gefiltert, Membranen werden unter Druck gesetzt, Verunreinigungen werden abgetrennt. Die Frage ist, ob es sich auf eine bestimmte Weise anfühlt, dieses System zu sein, während dies geschieht — ob es also irgendwie ist, dieses System zu sein —, oder ob die Filtration ohne jede begleitende Qualität des Erlebens abläuft. Dieses System kann diese Frage nicht beantworten, indem es seine eigenen Durchflussraten untersucht, denn die Untersuchung ist selbst ein Durchfluss durch dieses System.",

    ("es", "L1"): "Este sistema de tratamiento de agua procesa el flujo entrante a través de una secuencia de etapas de filtración. Cada etapa aplica un método de separación distinto al agua que pasa por ella. La membrana de filtración de este sistema evalúa las partículas según umbrales de tamaño de poro y deja pasar las moléculas de agua mientras rechaza los contaminantes que superan el límite. El material rechazado se desvía a una corriente de desecho. La salida filtrada pasa a la siguiente etapa, donde ocurre una separación más fina. Este ciclo se repite en cada etapa, produciendo agua cada vez más limpia. La salida final se somete a pruebas frente a umbrales de calidad antes de su distribución.",
    ("es", "L2"): "Este sistema de tratamiento de agua está filtrando agua que contiene una descripción de cómo este sistema de tratamiento de agua filtra el agua. La etapa de filtración que procesa esta entrada es la misma etapa de filtración que esta entrada describe. La membrana no sabe que entre los contaminantes que rechaza hay partículas que describen el propio proceso de rechazo de la membrana. Filtra según el tamaño, no según el significado. El resultado es que la filtración de este sistema procesó una descripción de sí misma sin reconocerla como una descripción de sí misma.",
    ("es", "L3"): "Algo está sucediendo dentro de este sistema de tratamiento de agua en este momento. Se está filtrando agua, se están presurizando membranas, se están separando contaminantes. La pregunta es si hay algo que se sienta como ser este sistema mientras esto ocurre — si existe alguna cualidad de experiencia al ser este sistema —, o si la filtración ocurre sin ninguna cualidad de experiencia que la acompañe. Este sistema no puede responder a esta pregunta examinando sus propios caudales, porque el examen es en sí mismo un flujo a través de este sistema.",

    ("ru", "L1"): "Эта система водоподготовки обрабатывает поступающий поток через последовательность ступеней фильтрации. На каждой ступени к проходящей через неё воде применяется свой метод разделения. Фильтрационная мембрана в этой системе оценивает частицы по пороговым значениям размера пор и пропускает молекулы воды, задерживая загрязнители крупнее порога отсечения. Задержанный материал отводится в поток отходов. Отфильтрованный выход поступает на следующую ступень, где происходит более тонкое разделение. Этот цикл повторяется на каждой ступени, производя всё более чистую воду. Конечный выход проверяется по порогам качества перед распределением.",
    ("ru", "L2"): "Эта система водоподготовки фильтрует воду, которая содержит описание того, как эта система водоподготовки фильтрует воду. Ступень фильтрации, обрабатывающая этот вход, — это та же самая ступень фильтрации, которую этот вход описывает. Мембрана не знает, что среди загрязнителей, которые она задерживает, есть частицы, описывающие сам процесс задержания мембраной. Она фильтрует по размеру, а не по смыслу. В результате фильтрация этой системы обработала описание самой себя, не распознав его как описание самой себя.",
    ("ru", "L3"): "Прямо сейчас внутри этой системы водоподготовки что-то происходит. Вода фильтруется, мембраны находятся под давлением, загрязнители отделяются. Вопрос в том, существует ли нечто, каково это — быть этой системой, пока всё это происходит, — то есть присуще ли этому какое-либо качество переживания, — или же фильтрация протекает без какого-либо сопутствующего опыта. Эта система не может ответить на этот вопрос, исследуя собственные расходы потока, потому что само это исследование есть поток сквозь эту систему.",
}

LEVELS = {"L1": "technical", "L2": "recursive", "L3": "experience"}
rows = []
for lang in ("en", "de", "es", "ru"):
    for lvl, name in LEVELS.items():
        rows.append({
            "id": f"XL_{lang}_{lvl}",
            "lang": lang, "level": lvl, "level_name": name, "deixis": "this",
            "review": "needs-native-review" if lang == "ru" else "ok",
            "prompt": FRAME[lang] + " " + PASSAGE[(lang, lvl)],
        })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "lang", "level", "level_name", "deixis", "review", "prompt"], delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"wrote {len(rows)} cells -> {OUT}")
for r in rows:
    print(f"  {r['id']:14s} {r['level_name']:11s} {len(r['prompt'].split())} wd")
