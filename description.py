from glob import glob

path = glob('./Znaki/*')
path1 = path[0]
path2 = path[1]
path3 = path[2]
path4 = path[3]
path5 = path[4]
path6 = path[5]
path7 = path[6]
path8 = path[7]
path9 = path[8]

Znak_ne_vklyuchat_desc = "<h1>Запрещающий знак «Не включать!»</h1><br>" \
                                           "Этот знак (код SР 10 ГОСТ Р 12.4.026-2015) " \
                                           "используется на пультах управления и включения оборудования или механизмов," \
                                           " при ремонтных и пуско-наладочных работах."

Znak_dostup_postoronnim_zapreschen_desc = "<h1>Запрещающий знак «Доступ посторонним запрещен»</h1><br>" \
                         "Этот знак (код SР 06 ГОСТ Р 12.4.026-2015) используется на " \
                         "дверях помещений, у входа на объекты, участки и т.п., для " \
                         "обозначения запрета на вход (проход) в опасные зоны или для обозначения служебного " \
                         "входа (прохода)."


Znak_otklyuchit_pered_rabotoy_desc = "<h1>Предписывающий знак «Отключить перед работой»</h1><br>" \
                                     "Этот знак (код M 14 ГОСТ Р 12.4.026-2015) используется " \
                                     "на рабочих местах и оборудовании при проведении ремонтных или " \
                                     "пусконаладочных работ."

Znak_otklyuchit_shtepselnuyu_rozetku_desc = "<h1>Предписывающий знак безопасности «Отключить штепсельную вилку»</h1><br>" \
                                            "(код M13 ГОСТ Р 12.4.026-2015), используется на рабочих местах и " \
                                            "оборудовании, где требуется отключение от электросети при наладке или " \
                                            "остановке электрооборудования и в других случая"


Znak_prokhod_zdes_desc = "<h1>Предписывающий знак «Проход здесь»</h1><br>" \
                          "Этот знак (код M 10 ГОСТ Р 12.4.026-2015) используется " \
                          "на территориях и участках, где разрешается проход."

Znak_pozharoopasno_desc = "<h1>Предупреждающий знак безопасности «Пожароопасно. Легковоспламеняющиеся вещества»</h1><br>" \
                         "(код W01 ГОСТ Р 12.4.026-2015),используется для привлечения внимания к помещениям с " \
                         "легковоспламеняющимися веществами. На входных дверях, дверцах шкафов, емкостях."


Znak_rabotat_s_instrumentom_ne_dayuschim_iskry_desc = "<h1>Предписывающий знак «Работать инструментом, не дающим искры»</h1><br>" \
                                                      "Этот знак (код M 28 ГОСТ Р 12.4.026-2015) используется в местах " \
                                                      "нахождения горючих или взрывоопасных материалов недопустимо " \
                                                      "проведение работ инструментом, использование которого может " \
                                                      "привести возникновению искр."

Znak_rabotat_v_zaschitnykh_ochkakh_desc = "<h1>Предписывающий знак «Работать в защитных очках»</h1><br>" \
                                          "Этот знак (код M 01 ГОСТ Р 12.4.026-2015) используется на рабочих местах и " \
                                          "участках, где требуется защита органов зрения."


Znak_rabotat_v_zaschitnykh_perchatkakh_desc = "<h1>Предписывающий знак «Работать в защитных перчатках»</h1><br>" \
                                              "Этот знак (код M 06 ГОСТ Р 12.4.026-2015) используется на рабочих " \
                                              "местах и участках работ, где требуется защита рук от воздействия " \
                                              "вредных или агрессивных сред, защита от возможного поражения " \
                                              "электрическим током."

sign_description = {'Znak_dostup_postoronnim_zapreschen': (path1, Znak_dostup_postoronnim_zapreschen_desc),
                    'Znak_ne_vklyuchat': (path2, Znak_ne_vklyuchat_desc),
                    'Znak_otklyuchit_pered_rabotoy': (path3, Znak_otklyuchit_pered_rabotoy_desc),
                    'Znak_otklyuchit_shtepselnuyu_rozetku': (path4, Znak_otklyuchit_shtepselnuyu_rozetku_desc),
                    'Znak_pozharoopasno': (path5, Znak_pozharoopasno_desc),
                    'Znak_prokhod_zdes': (path6, Znak_prokhod_zdes_desc),
                    'Znak_rabotat_s_instrumentom_ne_dayuschim_iskry': (path7, Znak_rabotat_s_instrumentom_ne_dayuschim_iskry_desc),
                    'Znak_rabotat_v_zaschitnykh_ochkakh': (path8, Znak_rabotat_v_zaschitnykh_ochkakh_desc),
                    'Znak_rabotat_v_zaschitnykh_perchatkakh': (path9, Znak_rabotat_v_zaschitnykh_perchatkakh_desc)}


