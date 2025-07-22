import discord
import asyncio

TOKEN = 'MTM0ODgyNDQyMTQ4OTU3MzkwOQ.GT0ad8.chdltb9kPJg1hXOWebqPGt83UhbAAjM76vJxPE'
CHANNEL_ID = 1354418108126855171

client = discord.Client()

@client.event
async def on_ready():
    channel = client.get_channel(CHANNEL_ID)
    await channel.send("ทดสอบส่งข้อความ")

asyncio.run(client.start(TOKEN))