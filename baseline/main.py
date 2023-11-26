from pyboy import PyBoy
pyboy = PyBoy('gb_ROM/PokemonRed.gb')
while not pyboy.tick():
    pass
pyboy.stop()